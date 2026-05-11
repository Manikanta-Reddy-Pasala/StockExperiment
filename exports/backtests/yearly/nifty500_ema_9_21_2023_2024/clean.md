# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 891.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 220 |
| ALERT1 | 139 |
| ALERT2 | 139 |
| ALERT2_SKIP | 72 |
| ALERT3 | 416 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 224 |
| PARTIAL | 36 |
| TARGET_HIT | 5 |
| STOP_HIT | 224 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 265 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 117 / 148
- **Target hits / Stop hits / Partials:** 5 / 224 / 36
- **Avg / median % per leg:** 0.79% / -0.34%
- **Sum % (uncompounded):** 208.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 84 | 29 | 34.5% | 3 | 81 | 0 | -0.13% | -11.3% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.50% | -4.5% |
| BUY @ 3rd Alert (retest2) | 81 | 29 | 35.8% | 3 | 78 | 0 | -0.08% | -6.8% |
| SELL (all) | 181 | 88 | 48.6% | 2 | 143 | 36 | 1.21% | 219.4% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.56% | -1.1% |
| SELL @ 3rd Alert (retest2) | 179 | 88 | 49.2% | 2 | 141 | 36 | 1.23% | 220.5% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.13% | -5.6% |
| retest2 (combined) | 260 | 117 | 45.0% | 5 | 219 | 36 | 0.82% | 213.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 14:15:00 | 1514.80 | 1521.59 | 1521.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 1475.45 | 1510.81 | 1516.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 13:15:00 | 1497.00 | 1496.05 | 1506.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 14:00:00 | 1497.00 | 1496.05 | 1506.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 1495.00 | 1495.75 | 1504.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 09:15:00 | 1483.80 | 1495.75 | 1504.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 11:00:00 | 1494.15 | 1497.74 | 1501.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-23 12:30:00 | 1489.30 | 1495.65 | 1499.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:15:00 | 1409.61 | 1430.95 | 1447.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:15:00 | 1419.44 | 1430.95 | 1447.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-29 09:15:00 | 1414.83 | 1430.95 | 1447.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-05-29 13:15:00 | 1428.00 | 1426.55 | 1439.55 | SL hit (close>ema200) qty=0.50 sl=1426.55 alert=retest2 |

### Cycle 2 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 11:15:00 | 1424.10 | 1416.41 | 1416.32 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 1406.25 | 1416.07 | 1416.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 13:15:00 | 1402.00 | 1410.19 | 1413.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-08 09:15:00 | 1405.75 | 1404.75 | 1408.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 1405.75 | 1404.75 | 1408.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 1405.75 | 1404.75 | 1408.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-08 11:15:00 | 1403.00 | 1404.80 | 1407.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 13:15:00 | 1415.35 | 1404.33 | 1406.49 | SL hit (close>static) qty=1.00 sl=1411.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 1384.00 | 1364.26 | 1363.96 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 13:15:00 | 1368.15 | 1378.03 | 1378.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 14:15:00 | 1363.80 | 1375.19 | 1377.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 1378.90 | 1374.94 | 1376.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 1378.90 | 1374.94 | 1376.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 1378.90 | 1374.94 | 1376.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 09:45:00 | 1378.00 | 1374.94 | 1376.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 1372.55 | 1374.46 | 1376.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:30:00 | 1377.50 | 1374.46 | 1376.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 12:15:00 | 1373.80 | 1374.28 | 1375.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 12:45:00 | 1374.05 | 1374.28 | 1375.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 1380.50 | 1375.52 | 1376.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 14:00:00 | 1380.50 | 1375.52 | 1376.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 14:15:00 | 1369.35 | 1374.29 | 1375.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:30:00 | 1363.00 | 1370.09 | 1373.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 11:00:00 | 1361.00 | 1360.70 | 1366.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 12:15:00 | 1364.80 | 1361.56 | 1366.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 12:45:00 | 1364.95 | 1362.12 | 1366.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 1345.40 | 1358.78 | 1364.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-26 15:15:00 | 1378.50 | 1363.76 | 1362.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 1378.50 | 1363.76 | 1362.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 14:15:00 | 1394.55 | 1382.86 | 1377.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 14:15:00 | 1402.00 | 1403.35 | 1395.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 14:45:00 | 1403.00 | 1403.35 | 1395.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 09:15:00 | 1391.15 | 1400.37 | 1395.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 10:00:00 | 1391.15 | 1400.37 | 1395.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 10:15:00 | 1389.40 | 1398.18 | 1394.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-04 11:00:00 | 1389.40 | 1398.18 | 1394.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2023-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 13:15:00 | 1386.30 | 1392.98 | 1393.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-05 10:15:00 | 1382.40 | 1389.57 | 1391.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1358.00 | 1355.37 | 1362.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1358.00 | 1355.37 | 1362.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1358.00 | 1355.37 | 1362.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-11 12:15:00 | 1348.20 | 1355.39 | 1361.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 13:15:00 | 1318.00 | 1316.05 | 1315.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2023-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 13:15:00 | 1318.00 | 1316.05 | 1315.97 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 1310.45 | 1316.28 | 1316.29 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 09:15:00 | 1332.40 | 1318.83 | 1317.23 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 10:15:00 | 1314.75 | 1316.64 | 1316.83 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 12:15:00 | 1320.05 | 1317.22 | 1317.05 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 13:15:00 | 1314.70 | 1316.72 | 1316.84 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 1319.00 | 1317.15 | 1317.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 1324.45 | 1318.61 | 1317.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 14:15:00 | 1317.20 | 1318.75 | 1318.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 14:15:00 | 1317.20 | 1318.75 | 1318.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1317.20 | 1318.75 | 1318.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 15:00:00 | 1317.20 | 1318.75 | 1318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1320.00 | 1319.00 | 1318.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 1313.70 | 1318.00 | 1317.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 10:15:00 | 1312.00 | 1316.80 | 1317.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 11:15:00 | 1309.25 | 1315.29 | 1316.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 12:15:00 | 1320.00 | 1316.23 | 1316.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 12:15:00 | 1320.00 | 1316.23 | 1316.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 1320.00 | 1316.23 | 1316.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-24 13:00:00 | 1320.00 | 1316.23 | 1316.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 1318.00 | 1316.59 | 1317.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-24 14:30:00 | 1313.45 | 1315.01 | 1316.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-31 09:15:00 | 1330.00 | 1300.93 | 1299.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1330.00 | 1300.93 | 1299.92 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 1300.00 | 1303.03 | 1303.30 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 09:15:00 | 1316.00 | 1305.41 | 1304.27 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 1293.00 | 1303.78 | 1305.17 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 15:15:00 | 1309.95 | 1304.23 | 1303.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 09:15:00 | 1338.00 | 1310.98 | 1306.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 09:15:00 | 1326.00 | 1326.54 | 1318.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-08 10:00:00 | 1326.00 | 1326.54 | 1318.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 09:15:00 | 1431.50 | 1420.12 | 1406.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:30:00 | 1417.00 | 1420.12 | 1406.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 13:15:00 | 1410.75 | 1418.38 | 1410.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-16 09:15:00 | 1417.30 | 1413.00 | 1409.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 11:15:00 | 1405.00 | 1412.58 | 1410.08 | SL hit (close<static) qty=1.00 sl=1408.55 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-16 13:15:00 | 1394.00 | 1406.05 | 1407.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 12:15:00 | 1388.35 | 1399.40 | 1403.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 11:15:00 | 1391.70 | 1380.82 | 1386.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 11:15:00 | 1391.70 | 1380.82 | 1386.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 1391.70 | 1380.82 | 1386.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 12:00:00 | 1391.70 | 1380.82 | 1386.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 1377.00 | 1380.05 | 1385.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 13:30:00 | 1371.85 | 1379.17 | 1384.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-21 14:15:00 | 1373.15 | 1379.17 | 1384.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-22 09:15:00 | 1399.65 | 1380.26 | 1383.38 | SL hit (close>static) qty=1.00 sl=1393.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 11:15:00 | 1411.00 | 1389.91 | 1387.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 1420.05 | 1395.94 | 1390.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 1405.50 | 1406.31 | 1399.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 12:45:00 | 1407.05 | 1406.31 | 1399.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 1402.65 | 1404.81 | 1400.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:45:00 | 1397.40 | 1404.81 | 1400.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 1403.00 | 1404.45 | 1400.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:30:00 | 1409.30 | 1406.19 | 1401.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 12:30:00 | 1408.20 | 1407.02 | 1403.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 14:15:00 | 1405.65 | 1406.14 | 1403.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 1409.00 | 1404.49 | 1402.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1410.05 | 1405.60 | 1403.49 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-25 11:15:00 | 1394.95 | 1402.17 | 1402.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 1394.95 | 1402.17 | 1402.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 13:15:00 | 1385.45 | 1397.99 | 1400.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1403.35 | 1398.02 | 1399.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1403.35 | 1398.02 | 1399.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1403.35 | 1398.02 | 1399.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 09:45:00 | 1405.20 | 1398.02 | 1399.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 1404.45 | 1399.31 | 1400.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:30:00 | 1404.00 | 1399.31 | 1400.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 1401.50 | 1399.75 | 1400.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:30:00 | 1401.75 | 1399.75 | 1400.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 13:15:00 | 1400.50 | 1399.86 | 1400.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 13:30:00 | 1398.90 | 1399.86 | 1400.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 14:15:00 | 1396.95 | 1399.28 | 1399.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 14:30:00 | 1401.55 | 1399.28 | 1399.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 09:15:00 | 1429.30 | 1405.24 | 1402.46 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 14:15:00 | 1412.00 | 1414.04 | 1414.26 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 09:15:00 | 1427.35 | 1416.37 | 1415.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 10:15:00 | 1465.75 | 1426.25 | 1419.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 14:15:00 | 1438.75 | 1440.56 | 1429.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-04 15:00:00 | 1438.75 | 1440.56 | 1429.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 1456.80 | 1470.36 | 1467.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 15:00:00 | 1456.80 | 1470.36 | 1467.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 15:15:00 | 1461.35 | 1468.56 | 1467.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 1495.05 | 1468.56 | 1467.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-12 10:30:00 | 1467.05 | 1478.60 | 1477.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 11:15:00 | 1453.25 | 1473.53 | 1475.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1453.25 | 1473.53 | 1475.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 12:15:00 | 1441.10 | 1467.04 | 1472.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 13:15:00 | 1448.00 | 1441.41 | 1452.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 13:45:00 | 1438.45 | 1441.41 | 1452.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 15:15:00 | 1455.00 | 1445.49 | 1452.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:15:00 | 1462.60 | 1445.49 | 1452.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1461.95 | 1448.78 | 1453.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:45:00 | 1466.00 | 1448.78 | 1453.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 14:15:00 | 1460.15 | 1455.95 | 1455.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 1487.05 | 1463.15 | 1459.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 1460.10 | 1468.39 | 1464.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 1460.10 | 1468.39 | 1464.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 1460.10 | 1468.39 | 1464.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 1460.10 | 1468.39 | 1464.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 1453.00 | 1465.31 | 1463.03 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 10:15:00 | 1449.35 | 1459.88 | 1460.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 11:15:00 | 1440.55 | 1450.48 | 1454.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 1444.00 | 1443.97 | 1449.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 1444.00 | 1443.97 | 1449.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 1444.00 | 1443.97 | 1449.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 12:30:00 | 1431.50 | 1440.73 | 1446.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:45:00 | 1421.15 | 1431.03 | 1439.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 14:15:00 | 1426.55 | 1419.55 | 1418.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 14:15:00 | 1426.55 | 1419.55 | 1418.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 15:15:00 | 1430.00 | 1421.64 | 1419.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 09:15:00 | 1419.45 | 1421.20 | 1419.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 09:15:00 | 1419.45 | 1421.20 | 1419.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 1419.45 | 1421.20 | 1419.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:30:00 | 1413.50 | 1421.20 | 1419.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 1415.95 | 1420.15 | 1419.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:00:00 | 1415.95 | 1420.15 | 1419.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 1416.75 | 1419.47 | 1419.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:30:00 | 1415.30 | 1419.47 | 1419.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 1411.75 | 1417.93 | 1418.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 1407.95 | 1415.93 | 1417.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 15:15:00 | 1419.00 | 1416.40 | 1417.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 15:15:00 | 1419.00 | 1416.40 | 1417.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 1419.00 | 1416.40 | 1417.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 1415.60 | 1416.40 | 1417.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 1414.85 | 1416.09 | 1417.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 15:00:00 | 1406.90 | 1414.36 | 1416.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 09:30:00 | 1409.90 | 1413.03 | 1415.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 12:15:00 | 1410.15 | 1413.08 | 1414.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 13:00:00 | 1409.05 | 1412.27 | 1414.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 09:15:00 | 1414.90 | 1411.25 | 1413.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:30:00 | 1418.90 | 1411.25 | 1413.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 1413.35 | 1411.67 | 1413.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 11:15:00 | 1408.65 | 1411.67 | 1413.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-04 14:15:00 | 1410.50 | 1410.89 | 1412.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-05 09:15:00 | 1417.00 | 1413.22 | 1413.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 1417.00 | 1413.22 | 1413.11 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 11:15:00 | 1411.40 | 1413.00 | 1413.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 1408.60 | 1412.12 | 1412.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 13:15:00 | 1412.15 | 1412.13 | 1412.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 13:15:00 | 1412.15 | 1412.13 | 1412.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 1412.15 | 1412.13 | 1412.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:00:00 | 1412.15 | 1412.13 | 1412.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 1410.00 | 1411.70 | 1412.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-05 14:30:00 | 1412.65 | 1411.70 | 1412.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 1421.50 | 1413.54 | 1413.07 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 15:15:00 | 1406.00 | 1414.22 | 1414.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1392.00 | 1409.78 | 1412.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 09:15:00 | 1391.15 | 1389.86 | 1398.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-10 14:15:00 | 1379.65 | 1385.79 | 1393.78 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 1390.50 | 1384.19 | 1390.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 09:30:00 | 1395.00 | 1384.19 | 1390.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1388.40 | 1385.03 | 1390.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:30:00 | 1391.50 | 1385.03 | 1390.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 1391.35 | 1386.29 | 1390.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-11 11:15:00 | 1391.35 | 1386.29 | 1390.64 | SL hit (close>ema400) qty=1.00 sl=1390.64 alert=retest1 |

### Cycle 36 — BUY (started 2023-10-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 10:15:00 | 1408.90 | 1394.52 | 1392.83 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 10:15:00 | 1389.75 | 1393.80 | 1394.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-16 12:15:00 | 1383.00 | 1390.55 | 1392.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 11:15:00 | 1386.80 | 1383.72 | 1387.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 11:15:00 | 1386.80 | 1383.72 | 1387.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 1386.80 | 1383.72 | 1387.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:00:00 | 1386.80 | 1383.72 | 1387.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 12:15:00 | 1383.30 | 1383.63 | 1387.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:45:00 | 1384.30 | 1383.63 | 1387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1373.35 | 1368.14 | 1374.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-18 15:00:00 | 1373.35 | 1368.14 | 1374.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 15:15:00 | 1380.05 | 1370.52 | 1375.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 09:15:00 | 1368.00 | 1370.52 | 1375.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 13:45:00 | 1371.10 | 1371.80 | 1374.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 14:30:00 | 1371.15 | 1371.74 | 1373.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 1367.85 | 1371.69 | 1373.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 1367.05 | 1370.76 | 1373.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 11:15:00 | 1364.30 | 1370.61 | 1372.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 1362.05 | 1367.68 | 1370.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1299.60 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1302.54 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1302.59 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1299.46 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1296.08 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1293.95 | 1326.59 | 1338.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 15:15:00 | 1320.00 | 1317.57 | 1327.68 | SL hit (close>ema200) qty=0.50 sl=1317.57 alert=retest2 |

### Cycle 38 — BUY (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 13:15:00 | 1340.00 | 1318.67 | 1315.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 09:15:00 | 1355.00 | 1334.05 | 1324.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 11:15:00 | 1359.15 | 1361.98 | 1348.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-03 12:00:00 | 1359.15 | 1361.98 | 1348.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1350.00 | 1359.58 | 1348.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:45:00 | 1350.10 | 1359.58 | 1348.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 13:15:00 | 1349.00 | 1357.47 | 1348.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 13:45:00 | 1348.40 | 1357.47 | 1348.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 14:15:00 | 1339.00 | 1353.77 | 1347.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 15:00:00 | 1339.00 | 1353.77 | 1347.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 1343.70 | 1351.76 | 1347.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 1345.20 | 1351.76 | 1347.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:45:00 | 1345.00 | 1351.56 | 1347.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-16 09:15:00 | 1367.75 | 1376.02 | 1377.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 09:15:00 | 1367.75 | 1376.02 | 1377.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 10:15:00 | 1364.15 | 1373.64 | 1375.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 10:15:00 | 1367.00 | 1365.83 | 1369.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-17 10:30:00 | 1366.35 | 1365.83 | 1369.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 1379.05 | 1368.48 | 1370.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:00:00 | 1379.05 | 1368.48 | 1370.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 12:15:00 | 1379.75 | 1370.73 | 1371.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 12:30:00 | 1379.00 | 1370.73 | 1371.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 14:15:00 | 1367.25 | 1369.60 | 1370.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-17 15:00:00 | 1367.25 | 1369.60 | 1370.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 1358.40 | 1366.24 | 1369.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 1362.50 | 1366.24 | 1369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 1347.15 | 1362.42 | 1367.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-20 10:45:00 | 1360.00 | 1362.42 | 1367.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1347.05 | 1344.00 | 1349.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 12:15:00 | 1340.00 | 1345.25 | 1349.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 1357.90 | 1349.05 | 1349.69 | SL hit (close>static) qty=1.00 sl=1353.25 alert=retest2 |

### Cycle 40 — BUY (started 2023-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 09:15:00 | 1354.95 | 1350.78 | 1350.31 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1346.05 | 1349.71 | 1349.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 13:15:00 | 1343.50 | 1348.47 | 1349.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 15:15:00 | 1349.00 | 1347.62 | 1348.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 15:15:00 | 1349.00 | 1347.62 | 1348.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 1349.00 | 1347.62 | 1348.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-28 09:15:00 | 1346.00 | 1347.62 | 1348.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 09:15:00 | 1347.05 | 1347.51 | 1348.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:15:00 | 1345.00 | 1347.51 | 1348.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 11:45:00 | 1345.00 | 1345.29 | 1346.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 12:45:00 | 1342.75 | 1345.29 | 1346.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 13:30:00 | 1343.85 | 1344.96 | 1345.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 14:15:00 | 1343.85 | 1344.74 | 1345.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 15:00:00 | 1343.85 | 1344.74 | 1345.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 1348.00 | 1345.39 | 1345.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-30 09:15:00 | 1345.75 | 1345.39 | 1345.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 1345.00 | 1345.31 | 1345.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 1409.15 | 1358.08 | 1351.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 10:15:00 | 1409.15 | 1358.08 | 1351.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 10:15:00 | 1451.30 | 1426.21 | 1399.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 15:15:00 | 1431.00 | 1431.87 | 1413.12 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 09:15:00 | 1450.90 | 1431.87 | 1413.12 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-05 10:00:00 | 1439.75 | 1433.45 | 1415.54 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 1418.75 | 1430.07 | 1417.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 1421.70 | 1430.07 | 1417.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 12:15:00 | 1414.00 | 1426.86 | 1416.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-12-05 12:15:00 | 1414.00 | 1426.86 | 1416.81 | SL hit (close<ema400) qty=1.00 sl=1416.81 alert=retest1 |

### Cycle 43 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 1416.40 | 1425.03 | 1425.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 1414.50 | 1422.92 | 1424.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1453.95 | 1418.20 | 1418.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1453.95 | 1418.20 | 1418.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1453.95 | 1418.20 | 1418.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:00:00 | 1453.95 | 1418.20 | 1418.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 1462.95 | 1427.15 | 1422.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 10:15:00 | 1492.95 | 1460.06 | 1444.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 1482.00 | 1504.73 | 1489.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 10:15:00 | 1482.00 | 1504.73 | 1489.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 1482.00 | 1504.73 | 1489.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:00:00 | 1482.00 | 1504.73 | 1489.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 1483.60 | 1500.50 | 1489.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 1500.00 | 1489.87 | 1487.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 13:15:00 | 1459.90 | 1498.07 | 1502.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 1459.90 | 1498.07 | 1502.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 1448.55 | 1488.16 | 1497.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 1472.55 | 1471.53 | 1482.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 1472.55 | 1471.53 | 1482.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1493.75 | 1476.05 | 1482.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 1493.75 | 1476.05 | 1482.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 1499.70 | 1480.78 | 1484.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:45:00 | 1504.10 | 1480.78 | 1484.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 13:15:00 | 1478.60 | 1483.45 | 1484.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 13:30:00 | 1486.05 | 1483.45 | 1484.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 1513.70 | 1489.50 | 1487.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 09:15:00 | 1557.90 | 1506.94 | 1495.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 13:15:00 | 1546.20 | 1548.25 | 1531.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-27 13:45:00 | 1544.60 | 1548.25 | 1531.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 10:15:00 | 1537.50 | 1544.19 | 1534.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 11:15:00 | 1541.90 | 1544.19 | 1534.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 12:15:00 | 1545.00 | 1543.55 | 1535.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 1550.95 | 1539.93 | 1536.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 10:15:00 | 1542.70 | 1539.69 | 1536.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 10:15:00 | 1550.00 | 1541.75 | 1537.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 11:30:00 | 1552.80 | 1543.78 | 1538.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 12:15:00 | 1553.55 | 1543.78 | 1538.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 13:00:00 | 1554.85 | 1545.99 | 1540.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 15:15:00 | 1554.00 | 1546.05 | 1541.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 1600.25 | 1589.39 | 1573.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 1587.65 | 1589.39 | 1573.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 12:15:00 | 1576.20 | 1588.24 | 1575.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 13:00:00 | 1576.20 | 1588.24 | 1575.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 1580.05 | 1586.61 | 1576.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-03 15:15:00 | 1570.00 | 1573.65 | 1573.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-03 15:15:00 | 1570.00 | 1573.65 | 1573.84 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-01-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-04 12:15:00 | 1574.40 | 1573.93 | 1573.90 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-01-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 13:15:00 | 1572.95 | 1573.73 | 1573.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-04 14:15:00 | 1565.05 | 1572.00 | 1573.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 15:15:00 | 1572.75 | 1572.15 | 1572.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 15:15:00 | 1572.75 | 1572.15 | 1572.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 1572.75 | 1572.15 | 1572.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:15:00 | 1579.95 | 1572.15 | 1572.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 1571.90 | 1572.10 | 1572.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 10:45:00 | 1566.60 | 1570.17 | 1571.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 12:15:00 | 1569.70 | 1570.15 | 1571.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 14:00:00 | 1567.25 | 1570.35 | 1571.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 09:15:00 | 1567.00 | 1570.74 | 1571.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1551.00 | 1566.79 | 1569.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-08 10:45:00 | 1524.95 | 1558.70 | 1565.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 09:15:00 | 1491.21 | 1516.55 | 1532.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 12:15:00 | 1488.27 | 1504.72 | 1522.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 12:15:00 | 1488.89 | 1504.72 | 1522.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 12:15:00 | 1488.65 | 1504.72 | 1522.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-11 09:15:00 | 1506.05 | 1501.07 | 1514.85 | SL hit (close>ema200) qty=0.50 sl=1501.07 alert=retest2 |

### Cycle 50 — BUY (started 2024-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 09:15:00 | 1529.00 | 1518.22 | 1517.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-12 10:15:00 | 1545.10 | 1523.60 | 1520.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 13:15:00 | 1523.05 | 1525.79 | 1522.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 13:15:00 | 1523.05 | 1525.79 | 1522.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 13:15:00 | 1523.05 | 1525.79 | 1522.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 14:00:00 | 1523.05 | 1525.79 | 1522.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 14:15:00 | 1527.40 | 1526.11 | 1522.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-12 15:15:00 | 1520.65 | 1526.11 | 1522.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 1520.65 | 1525.02 | 1522.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:15:00 | 1517.75 | 1525.02 | 1522.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1510.55 | 1522.13 | 1521.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:00:00 | 1510.55 | 1522.13 | 1521.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 10:15:00 | 1507.00 | 1519.10 | 1520.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 11:15:00 | 1504.90 | 1509.89 | 1513.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 15:15:00 | 1508.00 | 1505.06 | 1509.74 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 09:15:00 | 1492.85 | 1505.06 | 1509.74 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1485.20 | 1501.09 | 1507.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:30:00 | 1504.00 | 1501.09 | 1507.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 1488.40 | 1483.43 | 1491.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:30:00 | 1486.75 | 1483.43 | 1491.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 14:15:00 | 1487.00 | 1482.92 | 1489.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 15:00:00 | 1487.00 | 1482.92 | 1489.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 15:15:00 | 1480.00 | 1482.34 | 1488.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:15:00 | 1501.70 | 1482.34 | 1488.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1497.05 | 1485.28 | 1489.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 1497.05 | 1485.28 | 1489.16 | SL hit (close>ema400) qty=1.00 sl=1489.16 alert=retest1 |

### Cycle 52 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 1502.70 | 1491.51 | 1490.92 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 09:15:00 | 1485.05 | 1490.46 | 1490.56 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 10:15:00 | 1491.50 | 1490.67 | 1490.65 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 1486.80 | 1489.89 | 1490.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 13:15:00 | 1484.85 | 1488.51 | 1489.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 1447.60 | 1446.98 | 1457.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 1447.60 | 1446.98 | 1457.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1447.60 | 1446.98 | 1457.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 09:30:00 | 1451.35 | 1446.98 | 1457.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1436.70 | 1441.52 | 1449.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 10:30:00 | 1431.80 | 1439.37 | 1447.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 09:15:00 | 1458.10 | 1423.45 | 1429.12 | SL hit (close>static) qty=1.00 sl=1453.50 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 1492.90 | 1444.47 | 1438.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 1501.50 | 1455.88 | 1443.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 10:15:00 | 1471.65 | 1474.40 | 1459.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-01 11:00:00 | 1471.65 | 1474.40 | 1459.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 1465.50 | 1476.02 | 1470.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 13:45:00 | 1460.35 | 1476.02 | 1470.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 1465.05 | 1473.83 | 1469.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 15:00:00 | 1465.05 | 1473.83 | 1469.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 1469.00 | 1472.86 | 1469.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 1433.60 | 1472.86 | 1469.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2024-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 09:15:00 | 1432.00 | 1464.69 | 1466.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 10:15:00 | 1418.85 | 1455.52 | 1461.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1412.20 | 1409.15 | 1421.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1412.20 | 1409.15 | 1421.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1412.20 | 1409.15 | 1421.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 09:30:00 | 1396.70 | 1406.91 | 1414.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-08 10:30:00 | 1397.00 | 1403.88 | 1412.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-12 10:00:00 | 1400.00 | 1401.72 | 1403.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 10:45:00 | 1394.20 | 1396.33 | 1398.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1398.30 | 1394.03 | 1396.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 1398.30 | 1394.03 | 1396.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 1393.50 | 1393.92 | 1396.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 1390.00 | 1393.92 | 1396.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:45:00 | 1388.10 | 1392.03 | 1395.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-15 09:15:00 | 1401.00 | 1387.93 | 1390.48 | SL hit (close>static) qty=1.00 sl=1400.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 11:15:00 | 1399.00 | 1392.32 | 1392.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 1411.00 | 1397.17 | 1395.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 1402.65 | 1403.96 | 1399.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 14:15:00 | 1402.65 | 1403.96 | 1399.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1402.65 | 1403.96 | 1399.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:30:00 | 1402.95 | 1403.96 | 1399.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 1403.00 | 1403.77 | 1400.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 1408.00 | 1403.77 | 1400.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 09:45:00 | 1405.65 | 1403.64 | 1400.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 10:15:00 | 1406.20 | 1403.64 | 1400.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:15:00 | 1405.30 | 1403.86 | 1401.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1440.60 | 1448.84 | 1433.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:45:00 | 1430.00 | 1448.84 | 1433.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1449.55 | 1461.83 | 1453.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-26 09:45:00 | 1448.75 | 1461.83 | 1453.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 10:15:00 | 1455.55 | 1460.57 | 1453.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 11:15:00 | 1457.75 | 1460.57 | 1453.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 14:30:00 | 1467.00 | 1458.17 | 1454.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 10:30:00 | 1456.95 | 1459.37 | 1455.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 1457.60 | 1460.53 | 1457.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1459.65 | 1460.59 | 1457.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 15:00:00 | 1459.65 | 1460.59 | 1457.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 15:15:00 | 1462.00 | 1460.87 | 1458.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 09:15:00 | 1456.00 | 1460.87 | 1458.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1447.00 | 1458.10 | 1457.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 10:00:00 | 1447.00 | 1458.10 | 1457.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-28 10:15:00 | 1432.05 | 1452.89 | 1454.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1432.05 | 1452.89 | 1454.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 11:15:00 | 1427.40 | 1447.79 | 1452.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 1416.10 | 1408.39 | 1415.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 1416.10 | 1408.39 | 1415.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1416.10 | 1408.39 | 1415.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:30:00 | 1423.40 | 1408.39 | 1415.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1412.90 | 1409.29 | 1415.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 1420.95 | 1409.29 | 1415.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 1415.00 | 1410.43 | 1415.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:15:00 | 1406.80 | 1410.43 | 1415.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1400.20 | 1408.39 | 1413.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 10:45:00 | 1398.00 | 1406.31 | 1412.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 12:00:00 | 1398.70 | 1404.79 | 1411.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 12:30:00 | 1398.05 | 1402.66 | 1409.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 09:15:00 | 1393.20 | 1406.97 | 1407.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 1375.00 | 1400.58 | 1404.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-06 11:00:00 | 1371.00 | 1394.66 | 1401.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:30:00 | 1372.00 | 1375.45 | 1383.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 1328.10 | 1346.84 | 1362.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 1328.76 | 1346.84 | 1362.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 10:15:00 | 1328.15 | 1346.84 | 1362.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-12 11:15:00 | 1323.54 | 1341.79 | 1358.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-12 14:15:00 | 1335.95 | 1335.35 | 1350.87 | SL hit (close>ema200) qty=0.50 sl=1335.35 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 12:15:00 | 1312.80 | 1299.22 | 1299.13 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 14:15:00 | 1296.90 | 1299.64 | 1299.82 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 1302.00 | 1300.11 | 1300.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 1317.55 | 1303.60 | 1301.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 13:15:00 | 1303.10 | 1306.91 | 1304.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-22 13:15:00 | 1303.10 | 1306.91 | 1304.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 13:15:00 | 1303.10 | 1306.91 | 1304.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 13:45:00 | 1302.85 | 1306.91 | 1304.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 14:15:00 | 1305.00 | 1306.53 | 1304.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 15:00:00 | 1305.00 | 1306.53 | 1304.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 15:15:00 | 1315.60 | 1308.35 | 1305.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:15:00 | 1310.10 | 1308.35 | 1305.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1298.40 | 1306.36 | 1304.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 10:00:00 | 1298.40 | 1306.36 | 1304.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 10:15:00 | 1296.50 | 1304.39 | 1303.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:00:00 | 1296.50 | 1304.39 | 1303.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 11:15:00 | 1292.25 | 1301.96 | 1302.86 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 13:15:00 | 1316.15 | 1305.00 | 1304.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-27 10:15:00 | 1324.00 | 1311.14 | 1307.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 10:15:00 | 1332.00 | 1334.80 | 1323.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 11:00:00 | 1332.00 | 1334.80 | 1323.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 1326.10 | 1333.06 | 1324.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 11:45:00 | 1324.05 | 1333.06 | 1324.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 12:15:00 | 1326.65 | 1331.78 | 1324.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 12:45:00 | 1325.55 | 1331.78 | 1324.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 13:15:00 | 1334.00 | 1332.22 | 1325.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 14:15:00 | 1343.20 | 1332.22 | 1325.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 15:00:00 | 1341.80 | 1334.14 | 1326.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-01 09:15:00 | 1349.95 | 1332.31 | 1326.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 14:00:00 | 1340.15 | 1345.24 | 1340.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 1328.00 | 1339.03 | 1338.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 1332.00 | 1339.03 | 1338.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 1337.00 | 1338.00 | 1337.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:30:00 | 1334.80 | 1338.00 | 1337.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1342.00 | 1342.49 | 1340.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:00:00 | 1342.00 | 1342.49 | 1340.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 1340.60 | 1342.11 | 1340.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 1339.85 | 1342.11 | 1340.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 1340.00 | 1341.69 | 1340.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 11:30:00 | 1339.40 | 1341.69 | 1340.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 1340.00 | 1341.35 | 1340.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:45:00 | 1340.35 | 1341.35 | 1340.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 13:15:00 | 1339.00 | 1340.88 | 1340.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-04 14:15:00 | 1335.00 | 1339.70 | 1340.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 1335.00 | 1339.70 | 1340.04 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 09:15:00 | 1347.00 | 1341.49 | 1340.81 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 12:15:00 | 1337.05 | 1342.56 | 1342.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 09:15:00 | 1329.25 | 1337.02 | 1339.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 1334.10 | 1329.66 | 1333.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 1334.10 | 1329.66 | 1333.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 1334.10 | 1329.66 | 1333.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:45:00 | 1336.95 | 1329.66 | 1333.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 1334.05 | 1330.54 | 1333.93 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 1354.20 | 1337.30 | 1336.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 09:15:00 | 1370.40 | 1352.23 | 1344.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 14:15:00 | 1356.55 | 1359.08 | 1351.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 15:00:00 | 1356.55 | 1359.08 | 1351.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 15:15:00 | 1367.00 | 1360.66 | 1353.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:15:00 | 1335.90 | 1360.66 | 1353.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1349.00 | 1358.33 | 1352.73 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 12:15:00 | 1328.55 | 1347.58 | 1348.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 13:15:00 | 1320.45 | 1342.15 | 1346.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1335.00 | 1334.69 | 1341.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 09:45:00 | 1338.90 | 1334.69 | 1341.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 1337.50 | 1335.97 | 1340.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:30:00 | 1338.50 | 1335.97 | 1340.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 13:15:00 | 1336.45 | 1336.06 | 1339.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 13:30:00 | 1342.00 | 1336.06 | 1339.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 1339.60 | 1336.77 | 1339.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 1332.05 | 1336.82 | 1339.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 12:15:00 | 1331.85 | 1337.65 | 1339.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 12:15:00 | 1314.50 | 1308.62 | 1307.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 1314.50 | 1308.62 | 1307.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 15:15:00 | 1316.00 | 1311.05 | 1309.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 11:15:00 | 1306.60 | 1311.99 | 1310.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 11:15:00 | 1306.60 | 1311.99 | 1310.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1306.60 | 1311.99 | 1310.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 09:15:00 | 1319.05 | 1312.40 | 1310.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:45:00 | 1318.95 | 1313.75 | 1311.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:15:00 | 1321.80 | 1313.75 | 1311.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 11:45:00 | 1320.95 | 1314.84 | 1312.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1325.00 | 1320.76 | 1316.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 1328.55 | 1320.76 | 1316.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 12:15:00 | 1327.40 | 1324.43 | 1319.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 13:30:00 | 1327.35 | 1324.53 | 1320.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 10:15:00 | 1313.50 | 1320.51 | 1319.63 | SL hit (close<static) qty=1.00 sl=1314.80 alert=retest2 |

### Cycle 71 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 1309.15 | 1318.24 | 1318.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 14:15:00 | 1302.50 | 1309.34 | 1312.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 1311.00 | 1308.64 | 1311.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 10:15:00 | 1311.00 | 1308.64 | 1311.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1311.00 | 1308.64 | 1311.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-07 09:15:00 | 1302.45 | 1306.53 | 1309.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 1304.90 | 1291.99 | 1290.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1304.90 | 1291.99 | 1290.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 14:15:00 | 1325.80 | 1307.58 | 1299.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1342.45 | 1342.95 | 1327.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:45:00 | 1340.55 | 1342.95 | 1327.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 1338.65 | 1341.17 | 1334.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:30:00 | 1335.00 | 1341.17 | 1334.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1325.15 | 1338.06 | 1334.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1325.15 | 1338.06 | 1334.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1328.05 | 1336.06 | 1333.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:30:00 | 1325.75 | 1336.06 | 1333.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1334.00 | 1335.71 | 1334.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 1346.60 | 1335.39 | 1334.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1354.95 | 1376.72 | 1377.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1354.95 | 1376.72 | 1377.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 1341.40 | 1350.05 | 1358.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 1338.80 | 1337.73 | 1346.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 1334.05 | 1337.73 | 1346.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1320.50 | 1321.72 | 1331.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:15:00 | 1306.00 | 1318.77 | 1329.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:45:00 | 1305.85 | 1312.82 | 1323.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 10:30:00 | 1298.05 | 1304.42 | 1316.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 11:30:00 | 1302.60 | 1294.04 | 1302.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1301.15 | 1295.46 | 1302.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 1301.15 | 1295.46 | 1302.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1298.00 | 1295.97 | 1301.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:30:00 | 1300.80 | 1295.97 | 1301.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1299.00 | 1296.58 | 1301.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 1303.95 | 1296.58 | 1301.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1309.00 | 1298.87 | 1301.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 1313.55 | 1298.87 | 1301.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1311.25 | 1301.35 | 1302.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1310.40 | 1301.35 | 1302.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-06 11:15:00 | 1313.90 | 1303.86 | 1303.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 1313.90 | 1303.86 | 1303.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 1318.95 | 1308.90 | 1306.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1324.00 | 1328.45 | 1320.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 14:15:00 | 1324.00 | 1328.45 | 1320.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1324.00 | 1328.45 | 1320.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:45:00 | 1323.80 | 1328.45 | 1320.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1366.30 | 1370.31 | 1365.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 1367.55 | 1370.31 | 1365.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1359.25 | 1368.10 | 1364.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 1359.25 | 1368.10 | 1364.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 11:15:00 | 1355.00 | 1365.48 | 1363.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:45:00 | 1355.00 | 1365.48 | 1363.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 13:15:00 | 1351.05 | 1360.68 | 1361.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 14:15:00 | 1350.00 | 1358.55 | 1360.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 1353.40 | 1351.82 | 1356.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 1353.40 | 1351.82 | 1356.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 1353.40 | 1351.82 | 1356.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 1353.40 | 1351.82 | 1356.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 1358.35 | 1353.13 | 1356.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:00:00 | 1358.35 | 1353.13 | 1356.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1368.00 | 1356.10 | 1357.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 13:30:00 | 1363.25 | 1356.10 | 1357.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 1364.40 | 1357.76 | 1358.11 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 1364.00 | 1359.01 | 1358.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 1394.00 | 1366.01 | 1361.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 15:15:00 | 1450.00 | 1450.29 | 1432.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:15:00 | 1427.75 | 1450.29 | 1432.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1445.25 | 1449.29 | 1433.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:30:00 | 1427.00 | 1449.29 | 1433.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1435.80 | 1446.59 | 1434.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:45:00 | 1427.80 | 1446.59 | 1434.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1450.05 | 1447.28 | 1435.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 1437.40 | 1447.28 | 1435.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 1440.60 | 1452.29 | 1444.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:00:00 | 1440.60 | 1452.29 | 1444.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1446.90 | 1451.21 | 1444.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 1457.00 | 1451.21 | 1444.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 1451.90 | 1451.17 | 1445.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 10:45:00 | 1451.70 | 1450.11 | 1447.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 1438.05 | 1445.77 | 1445.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1438.05 | 1445.77 | 1445.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 1433.00 | 1443.21 | 1444.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 12:15:00 | 1440.00 | 1438.90 | 1441.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 12:15:00 | 1440.00 | 1438.90 | 1441.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1440.00 | 1438.90 | 1441.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:45:00 | 1425.60 | 1432.23 | 1438.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 1481.50 | 1435.27 | 1432.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1481.50 | 1435.27 | 1432.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 1491.10 | 1458.24 | 1444.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 1494.00 | 1499.64 | 1482.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 11:00:00 | 1494.00 | 1499.64 | 1482.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1486.65 | 1496.88 | 1488.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:45:00 | 1485.50 | 1496.88 | 1488.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1481.15 | 1493.73 | 1488.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:45:00 | 1473.10 | 1493.73 | 1488.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 1500.25 | 1495.04 | 1489.11 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 1479.90 | 1486.93 | 1487.29 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 1497.00 | 1488.94 | 1488.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 13:15:00 | 1521.05 | 1496.33 | 1491.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 1489.35 | 1498.60 | 1494.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1489.35 | 1498.60 | 1494.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1489.35 | 1498.60 | 1494.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1487.50 | 1498.60 | 1494.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1482.75 | 1495.43 | 1493.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1482.75 | 1495.43 | 1493.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1485.25 | 1493.40 | 1492.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:30:00 | 1480.25 | 1493.40 | 1492.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 1502.15 | 1495.15 | 1493.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 13:15:00 | 1510.05 | 1495.15 | 1493.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 12:15:00 | 1485.25 | 1494.54 | 1495.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 1485.25 | 1494.54 | 1495.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 1462.70 | 1487.07 | 1491.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 13:15:00 | 1500.05 | 1466.19 | 1471.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 13:15:00 | 1500.05 | 1466.19 | 1471.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1500.05 | 1466.19 | 1471.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 1500.05 | 1466.19 | 1471.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1500.25 | 1473.00 | 1474.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 1502.05 | 1473.00 | 1474.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 15:15:00 | 1500.00 | 1478.40 | 1476.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 1506.25 | 1483.97 | 1479.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 1490.00 | 1490.90 | 1485.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-12 14:45:00 | 1491.15 | 1490.90 | 1485.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1480.75 | 1488.41 | 1485.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 09:30:00 | 1473.30 | 1488.41 | 1485.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 1476.95 | 1486.12 | 1484.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:00:00 | 1476.95 | 1486.12 | 1484.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1486.45 | 1486.18 | 1484.47 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 1475.90 | 1482.69 | 1483.19 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 13:15:00 | 1487.60 | 1482.62 | 1482.61 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 14:15:00 | 1477.90 | 1481.68 | 1482.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 15:15:00 | 1475.05 | 1480.35 | 1481.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 12:15:00 | 1472.05 | 1469.65 | 1475.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 12:15:00 | 1472.05 | 1469.65 | 1475.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 1472.05 | 1469.65 | 1475.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 1476.00 | 1469.65 | 1475.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 1490.35 | 1473.79 | 1476.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 1490.35 | 1473.79 | 1476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 1478.00 | 1474.63 | 1476.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 1459.10 | 1475.91 | 1476.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 13:30:00 | 1474.10 | 1455.29 | 1459.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 1457.05 | 1450.70 | 1453.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 10:15:00 | 1460.85 | 1455.99 | 1455.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1460.85 | 1455.99 | 1455.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1470.00 | 1460.72 | 1458.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 1526.90 | 1527.83 | 1510.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 1526.90 | 1527.83 | 1510.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 1509.95 | 1521.93 | 1514.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:45:00 | 1507.70 | 1521.93 | 1514.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 1516.00 | 1520.74 | 1514.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 1525.10 | 1519.19 | 1514.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-01 09:15:00 | 1677.61 | 1628.04 | 1591.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 1593.00 | 1607.07 | 1608.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 15:15:00 | 1585.00 | 1602.65 | 1606.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1627.35 | 1607.59 | 1608.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1627.35 | 1607.59 | 1608.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1627.35 | 1607.59 | 1608.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1594.10 | 1601.13 | 1605.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:30:00 | 1598.95 | 1581.15 | 1590.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 11:30:00 | 1598.80 | 1584.58 | 1590.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 1599.15 | 1586.49 | 1591.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1591.00 | 1587.40 | 1591.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1593.75 | 1587.40 | 1591.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1593.55 | 1588.63 | 1591.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 1596.85 | 1588.63 | 1591.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1594.90 | 1589.88 | 1591.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 1602.75 | 1589.88 | 1591.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-08 09:15:00 | 1625.00 | 1596.90 | 1594.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 1625.00 | 1596.90 | 1594.66 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 10:15:00 | 1594.10 | 1596.70 | 1596.82 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 11:15:00 | 1602.00 | 1597.76 | 1597.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 1640.75 | 1608.71 | 1602.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 1615.55 | 1629.22 | 1621.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 11:15:00 | 1615.55 | 1629.22 | 1621.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1615.55 | 1629.22 | 1621.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1615.55 | 1629.22 | 1621.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1602.00 | 1623.78 | 1619.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1598.35 | 1623.78 | 1619.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 1592.20 | 1614.34 | 1615.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1578.15 | 1603.69 | 1610.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 1609.80 | 1604.91 | 1610.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 10:15:00 | 1609.80 | 1604.91 | 1610.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 1609.80 | 1604.91 | 1610.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 1609.80 | 1604.91 | 1610.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 1599.95 | 1603.92 | 1609.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 12:30:00 | 1592.55 | 1603.14 | 1608.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 13:45:00 | 1594.60 | 1601.20 | 1607.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 10:00:00 | 1585.00 | 1572.27 | 1573.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 10:15:00 | 1514.87 | 1547.82 | 1558.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-26 13:15:00 | 1512.92 | 1534.84 | 1549.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 1545.00 | 1532.12 | 1543.83 | SL hit (close>ema200) qty=0.50 sl=1532.12 alert=retest2 |

### Cycle 92 — BUY (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 10:15:00 | 1516.30 | 1490.21 | 1487.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 12:15:00 | 1528.05 | 1514.68 | 1508.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1566.95 | 1570.19 | 1553.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 1566.95 | 1570.19 | 1553.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1562.00 | 1568.08 | 1555.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 1550.85 | 1564.47 | 1554.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 1571.00 | 1565.77 | 1556.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 13:45:00 | 1574.40 | 1568.21 | 1559.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 15:00:00 | 1574.40 | 1569.44 | 1561.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 09:45:00 | 1575.55 | 1571.60 | 1563.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 14:45:00 | 1576.50 | 1572.03 | 1567.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1565.60 | 1571.50 | 1568.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:45:00 | 1564.20 | 1571.50 | 1568.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1592.00 | 1575.60 | 1570.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 1573.85 | 1575.60 | 1570.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1567.00 | 1579.72 | 1575.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 1567.00 | 1579.72 | 1575.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1567.40 | 1577.25 | 1574.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:15:00 | 1574.90 | 1577.25 | 1574.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 1575.50 | 1576.90 | 1574.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:45:00 | 1579.90 | 1577.75 | 1575.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 14:45:00 | 1580.00 | 1578.81 | 1576.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:15:00 | 1580.70 | 1577.51 | 1575.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 1557.10 | 1573.59 | 1574.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1557.10 | 1573.59 | 1574.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 1529.85 | 1559.23 | 1567.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 12:15:00 | 1554.90 | 1551.55 | 1560.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 12:30:00 | 1555.90 | 1551.55 | 1560.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1564.95 | 1554.71 | 1559.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 1564.95 | 1554.71 | 1559.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1565.30 | 1556.83 | 1560.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 11:15:00 | 1551.80 | 1556.83 | 1560.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:45:00 | 1559.55 | 1556.52 | 1559.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 14:15:00 | 1559.95 | 1534.49 | 1533.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 1559.95 | 1534.49 | 1533.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 1581.00 | 1555.53 | 1547.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 1606.05 | 1613.24 | 1593.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:30:00 | 1609.25 | 1613.24 | 1593.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1587.25 | 1605.92 | 1593.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 14:00:00 | 1587.25 | 1605.92 | 1593.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1596.20 | 1603.98 | 1594.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 15:15:00 | 1594.00 | 1603.98 | 1594.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1594.00 | 1601.98 | 1594.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1555.95 | 1601.98 | 1594.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1597.50 | 1601.09 | 1594.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1579.20 | 1601.09 | 1594.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1597.90 | 1600.40 | 1595.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:30:00 | 1595.70 | 1600.40 | 1595.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1588.80 | 1598.08 | 1594.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 1588.80 | 1598.08 | 1594.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1579.90 | 1594.45 | 1593.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:30:00 | 1575.00 | 1594.45 | 1593.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 1582.20 | 1592.00 | 1592.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1528.20 | 1577.31 | 1585.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1526.80 | 1523.98 | 1546.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 1526.80 | 1523.98 | 1546.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 1544.75 | 1530.93 | 1542.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 1544.75 | 1530.93 | 1542.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1545.00 | 1533.74 | 1542.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1545.75 | 1533.74 | 1542.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1549.50 | 1536.89 | 1543.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 1549.70 | 1536.89 | 1543.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1557.75 | 1541.06 | 1544.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 1557.15 | 1541.06 | 1544.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1554.05 | 1543.66 | 1545.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:45:00 | 1554.05 | 1543.66 | 1545.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1559.65 | 1549.30 | 1548.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 10:15:00 | 1568.80 | 1558.88 | 1555.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 15:15:00 | 1565.40 | 1566.11 | 1560.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:15:00 | 1558.20 | 1566.11 | 1560.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1564.50 | 1565.79 | 1561.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:15:00 | 1558.60 | 1565.79 | 1561.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1567.90 | 1566.21 | 1561.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:45:00 | 1584.10 | 1570.71 | 1564.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 12:15:00 | 1595.00 | 1603.54 | 1604.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 1595.00 | 1603.54 | 1604.59 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-10-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 13:15:00 | 1615.75 | 1605.98 | 1605.60 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 15:15:00 | 1590.00 | 1603.52 | 1604.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 1586.80 | 1600.17 | 1602.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 1602.00 | 1600.54 | 1602.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 1602.00 | 1600.54 | 1602.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1602.00 | 1600.54 | 1602.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 1602.00 | 1600.54 | 1602.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1590.95 | 1598.62 | 1601.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:15:00 | 1573.80 | 1599.24 | 1601.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1495.11 | 1544.67 | 1566.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 1529.40 | 1526.83 | 1542.91 | SL hit (close>ema200) qty=0.50 sl=1526.83 alert=retest2 |

### Cycle 100 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 1504.70 | 1488.31 | 1486.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 1511.00 | 1500.34 | 1493.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1504.25 | 1518.33 | 1507.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1504.25 | 1518.33 | 1507.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1504.25 | 1518.33 | 1507.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1504.25 | 1518.33 | 1507.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1502.05 | 1515.07 | 1507.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1502.05 | 1515.07 | 1507.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1502.15 | 1512.49 | 1506.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:15:00 | 1499.50 | 1512.49 | 1506.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 1500.95 | 1510.18 | 1506.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:30:00 | 1499.60 | 1510.18 | 1506.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1505.55 | 1506.88 | 1505.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 1511.80 | 1506.48 | 1505.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 1525.50 | 1507.42 | 1506.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 13:15:00 | 1478.95 | 1525.50 | 1523.37 | SL hit (close<static) qty=1.00 sl=1502.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 1484.95 | 1517.39 | 1519.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1459.00 | 1494.76 | 1508.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 14:15:00 | 1485.00 | 1482.78 | 1497.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 15:00:00 | 1485.00 | 1482.78 | 1497.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1292.00 | 1286.14 | 1295.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 1292.00 | 1286.14 | 1295.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1295.25 | 1287.96 | 1295.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:00:00 | 1295.25 | 1287.96 | 1295.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1294.50 | 1289.27 | 1295.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:30:00 | 1299.15 | 1289.27 | 1295.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1295.75 | 1290.57 | 1295.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-25 09:45:00 | 1291.35 | 1294.72 | 1296.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 1299.85 | 1295.74 | 1296.39 | SL hit (close>static) qty=1.00 sl=1298.90 alert=retest2 |

### Cycle 102 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 1304.85 | 1297.60 | 1296.61 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 1289.00 | 1294.83 | 1295.53 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 1305.00 | 1296.86 | 1296.39 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1291.00 | 1296.68 | 1296.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 1283.70 | 1293.12 | 1295.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 1295.85 | 1290.45 | 1292.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 10:15:00 | 1295.85 | 1290.45 | 1292.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1295.85 | 1290.45 | 1292.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1295.85 | 1290.45 | 1292.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1295.65 | 1291.49 | 1292.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 1297.65 | 1291.49 | 1292.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 12:15:00 | 1298.60 | 1292.91 | 1293.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:00:00 | 1298.60 | 1292.91 | 1293.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-11-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 13:15:00 | 1297.15 | 1293.76 | 1293.67 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 14:15:00 | 1284.90 | 1291.99 | 1292.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 15:15:00 | 1282.00 | 1289.99 | 1291.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1300.00 | 1284.72 | 1286.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 1300.00 | 1284.72 | 1286.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1300.00 | 1284.72 | 1286.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 1299.00 | 1284.72 | 1286.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1296.80 | 1287.14 | 1287.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:15:00 | 1295.00 | 1287.14 | 1287.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 1284.95 | 1286.79 | 1287.58 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 1290.50 | 1288.09 | 1288.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 1296.55 | 1289.78 | 1288.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 12:15:00 | 1290.35 | 1291.28 | 1289.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 12:15:00 | 1290.35 | 1291.28 | 1289.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 1290.35 | 1291.28 | 1289.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:45:00 | 1290.75 | 1291.28 | 1289.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1289.65 | 1290.96 | 1289.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:45:00 | 1288.65 | 1290.96 | 1289.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1288.00 | 1290.37 | 1289.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:30:00 | 1288.05 | 1290.37 | 1289.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 1284.40 | 1289.17 | 1289.20 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 10:15:00 | 1293.50 | 1289.64 | 1289.38 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 14:15:00 | 1287.10 | 1289.37 | 1289.40 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 1291.20 | 1289.62 | 1289.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1294.30 | 1291.16 | 1290.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 1290.10 | 1291.31 | 1290.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 15:15:00 | 1290.10 | 1291.31 | 1290.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1290.10 | 1291.31 | 1290.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 1298.40 | 1291.31 | 1290.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 12:15:00 | 1288.00 | 1291.08 | 1290.75 | SL hit (close<static) qty=1.00 sl=1290.10 alert=retest2 |

### Cycle 113 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 1288.20 | 1290.50 | 1290.52 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 1337.90 | 1299.50 | 1294.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1354.75 | 1310.55 | 1300.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 14:15:00 | 1412.05 | 1421.13 | 1397.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 14:30:00 | 1412.75 | 1421.13 | 1397.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1405.70 | 1416.42 | 1399.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:30:00 | 1412.10 | 1416.42 | 1399.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 1416.85 | 1416.51 | 1400.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:45:00 | 1423.85 | 1417.67 | 1402.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:15:00 | 1421.00 | 1415.73 | 1409.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:45:00 | 1434.80 | 1426.47 | 1418.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 09:15:00 | 1424.55 | 1454.40 | 1455.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 1424.55 | 1454.40 | 1455.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 1406.75 | 1435.76 | 1443.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 1406.30 | 1405.89 | 1419.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 11:45:00 | 1407.85 | 1405.89 | 1419.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 1422.20 | 1409.15 | 1419.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 1422.20 | 1409.15 | 1419.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 1417.40 | 1410.80 | 1419.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1407.80 | 1414.80 | 1419.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:00:00 | 1414.00 | 1415.68 | 1418.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:00:00 | 1414.95 | 1415.53 | 1418.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 1416.25 | 1408.04 | 1413.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1433.60 | 1413.15 | 1415.15 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-31 10:15:00 | 1433.60 | 1413.15 | 1415.15 | SL hit (close>static) qty=1.00 sl=1425.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1425.05 | 1418.07 | 1417.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 1483.10 | 1432.96 | 1424.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 1515.00 | 1515.36 | 1499.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 1515.00 | 1515.36 | 1499.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1495.60 | 1510.07 | 1499.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1495.60 | 1510.07 | 1499.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1493.90 | 1506.84 | 1499.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1478.10 | 1506.84 | 1499.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1469.35 | 1498.53 | 1496.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1469.35 | 1498.53 | 1496.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1468.60 | 1492.54 | 1494.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1447.85 | 1478.35 | 1486.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1470.55 | 1434.03 | 1446.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1470.55 | 1434.03 | 1446.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1470.55 | 1434.03 | 1446.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 1494.45 | 1434.03 | 1446.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1463.10 | 1439.84 | 1448.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:45:00 | 1449.30 | 1445.27 | 1449.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 14:45:00 | 1448.50 | 1444.89 | 1448.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 1376.83 | 1398.73 | 1415.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 13:15:00 | 1376.08 | 1398.73 | 1415.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-15 09:15:00 | 1377.75 | 1375.48 | 1389.29 | SL hit (close>ema200) qty=0.50 sl=1375.48 alert=retest2 |

### Cycle 118 — BUY (started 2025-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 12:15:00 | 1396.35 | 1387.77 | 1387.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1405.45 | 1394.80 | 1391.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 12:15:00 | 1426.10 | 1428.30 | 1420.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 13:00:00 | 1426.10 | 1428.30 | 1420.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1430.25 | 1430.99 | 1424.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 1427.20 | 1430.99 | 1424.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 1425.60 | 1429.91 | 1424.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 11:00:00 | 1425.60 | 1429.91 | 1424.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 1409.50 | 1425.83 | 1423.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 1409.50 | 1425.83 | 1423.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 1412.75 | 1423.21 | 1422.10 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 1412.10 | 1420.99 | 1421.19 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 14:15:00 | 1423.15 | 1421.42 | 1421.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1439.80 | 1425.35 | 1423.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 11:15:00 | 1427.00 | 1427.59 | 1424.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:00:00 | 1427.00 | 1427.59 | 1424.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1416.20 | 1428.83 | 1426.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 1416.20 | 1428.83 | 1426.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 1422.20 | 1427.51 | 1426.47 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 1415.25 | 1425.06 | 1425.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 1404.75 | 1420.99 | 1423.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1342.70 | 1334.62 | 1359.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 1342.70 | 1334.62 | 1359.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1350.00 | 1343.88 | 1356.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 1353.05 | 1343.88 | 1356.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1354.10 | 1345.92 | 1356.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 1350.00 | 1345.92 | 1356.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1355.25 | 1347.79 | 1356.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:30:00 | 1355.45 | 1347.79 | 1356.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1349.00 | 1348.03 | 1355.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 1354.85 | 1348.03 | 1355.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 1366.85 | 1347.34 | 1352.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:45:00 | 1375.05 | 1347.34 | 1352.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 1354.50 | 1348.77 | 1352.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:45:00 | 1349.05 | 1353.57 | 1354.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 14:15:00 | 1364.80 | 1355.82 | 1355.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 1364.80 | 1355.82 | 1355.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 15:15:00 | 1385.00 | 1361.66 | 1357.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1403.85 | 1414.47 | 1397.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 1403.85 | 1414.47 | 1397.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1398.35 | 1411.25 | 1397.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 1424.00 | 1416.73 | 1401.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 15:15:00 | 1463.00 | 1470.18 | 1470.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 15:15:00 | 1463.00 | 1470.18 | 1470.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 1450.05 | 1466.15 | 1468.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 1467.55 | 1460.28 | 1463.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 1467.55 | 1460.28 | 1463.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1467.55 | 1460.28 | 1463.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 1467.55 | 1460.28 | 1463.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1467.00 | 1461.62 | 1464.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1439.35 | 1461.62 | 1464.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 11:15:00 | 1367.38 | 1406.53 | 1428.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 10:15:00 | 1295.41 | 1320.84 | 1342.27 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 124 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1321.05 | 1308.73 | 1307.52 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 12:15:00 | 1300.85 | 1310.04 | 1311.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 1299.20 | 1305.71 | 1308.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 1282.25 | 1282.05 | 1289.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:45:00 | 1281.05 | 1282.05 | 1289.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1288.15 | 1283.27 | 1289.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 1288.15 | 1283.27 | 1289.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1286.05 | 1283.83 | 1289.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 1279.30 | 1283.83 | 1289.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 1215.33 | 1244.94 | 1265.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 09:15:00 | 1151.37 | 1194.88 | 1226.51 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 126 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1248.25 | 1197.75 | 1192.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 10:15:00 | 1273.65 | 1243.21 | 1230.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1255.45 | 1260.57 | 1246.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1255.45 | 1260.57 | 1246.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1255.45 | 1260.57 | 1246.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 1259.95 | 1260.57 | 1246.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1244.00 | 1257.25 | 1246.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1244.00 | 1257.25 | 1246.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1235.00 | 1252.80 | 1245.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 1235.00 | 1252.80 | 1245.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1242.20 | 1250.68 | 1245.05 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1227.95 | 1240.49 | 1241.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 11:15:00 | 1220.00 | 1232.54 | 1237.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 1223.95 | 1223.54 | 1230.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 10:15:00 | 1216.90 | 1223.54 | 1230.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1219.35 | 1222.71 | 1229.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:15:00 | 1202.45 | 1222.71 | 1229.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 10:15:00 | 1216.50 | 1209.56 | 1208.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 1216.50 | 1209.56 | 1208.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 1264.00 | 1231.01 | 1223.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 11:15:00 | 1238.75 | 1246.54 | 1238.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:00:00 | 1238.75 | 1246.54 | 1238.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1236.15 | 1244.46 | 1238.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:00:00 | 1236.15 | 1244.46 | 1238.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 1235.05 | 1242.58 | 1238.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:00:00 | 1235.05 | 1242.58 | 1238.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 1234.60 | 1239.93 | 1237.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 1245.80 | 1239.93 | 1237.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 10:15:00 | 1231.90 | 1237.93 | 1237.07 | SL hit (close<static) qty=1.00 sl=1232.10 alert=retest2 |

### Cycle 129 — SELL (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 11:15:00 | 1220.50 | 1234.44 | 1235.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 09:15:00 | 1216.00 | 1225.81 | 1230.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 11:15:00 | 1211.70 | 1209.91 | 1217.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 12:00:00 | 1211.70 | 1209.91 | 1217.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1196.95 | 1199.10 | 1206.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:30:00 | 1206.50 | 1199.10 | 1206.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1207.00 | 1198.87 | 1204.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 1190.70 | 1197.51 | 1203.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:45:00 | 1188.50 | 1195.27 | 1201.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 13:30:00 | 1191.00 | 1189.84 | 1192.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 14:15:00 | 1217.00 | 1195.27 | 1194.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 14:15:00 | 1217.00 | 1195.27 | 1194.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 15:15:00 | 1226.50 | 1201.52 | 1197.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 1195.00 | 1200.21 | 1197.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 1195.00 | 1200.21 | 1197.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1195.00 | 1200.21 | 1197.37 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 1187.30 | 1199.75 | 1201.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1116.50 | 1178.05 | 1189.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 1166.45 | 1156.47 | 1172.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 1166.45 | 1156.47 | 1172.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 1185.00 | 1162.18 | 1173.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1161.10 | 1164.20 | 1172.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1158.30 | 1169.75 | 1172.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:45:00 | 1163.40 | 1163.01 | 1167.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 1179.50 | 1167.95 | 1167.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1179.50 | 1167.95 | 1167.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 1193.60 | 1178.07 | 1174.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1180.80 | 1181.05 | 1177.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 1180.80 | 1181.05 | 1177.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1180.80 | 1181.05 | 1177.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 13:15:00 | 1205.00 | 1184.32 | 1180.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1206.30 | 1194.40 | 1187.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 09:15:00 | 1205.60 | 1203.41 | 1195.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 1203.70 | 1199.96 | 1196.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1182.70 | 1214.53 | 1211.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1182.70 | 1214.53 | 1211.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 1181.30 | 1207.88 | 1208.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1181.30 | 1207.88 | 1208.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 1170.50 | 1182.15 | 1187.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 15:15:00 | 1180.60 | 1174.29 | 1179.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 15:15:00 | 1180.60 | 1174.29 | 1179.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1180.60 | 1174.29 | 1179.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 1178.70 | 1175.09 | 1179.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1182.00 | 1176.48 | 1179.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:15:00 | 1178.20 | 1178.95 | 1180.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 1187.20 | 1180.60 | 1181.19 | SL hit (close>static) qty=1.00 sl=1187.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 1188.70 | 1182.22 | 1181.87 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1173.00 | 1180.91 | 1181.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1151.30 | 1173.39 | 1178.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1173.40 | 1173.39 | 1177.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 1173.40 | 1173.39 | 1177.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1184.00 | 1175.51 | 1178.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 1184.80 | 1175.51 | 1178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1181.60 | 1176.73 | 1178.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 13:30:00 | 1179.30 | 1177.58 | 1178.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 14:15:00 | 1178.50 | 1177.58 | 1178.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 1223.00 | 1187.86 | 1183.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1223.00 | 1187.86 | 1183.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1239.00 | 1200.85 | 1195.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 1288.10 | 1289.53 | 1277.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 12:15:00 | 1277.60 | 1286.02 | 1278.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 1277.60 | 1286.02 | 1278.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 13:00:00 | 1277.60 | 1286.02 | 1278.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1261.80 | 1281.18 | 1276.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1261.80 | 1281.18 | 1276.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1277.00 | 1280.34 | 1276.72 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1268.20 | 1276.79 | 1276.92 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1279.10 | 1277.19 | 1277.00 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1272.70 | 1276.29 | 1276.61 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1281.00 | 1276.72 | 1276.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 1290.00 | 1282.51 | 1279.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 1423.20 | 1429.12 | 1416.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 1423.20 | 1429.12 | 1416.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1481.40 | 1439.62 | 1423.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 1548.10 | 1481.30 | 1459.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 13:15:00 | 1491.70 | 1503.50 | 1495.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 1491.20 | 1501.04 | 1494.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1496.10 | 1493.11 | 1492.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1493.50 | 1493.48 | 1492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 1493.50 | 1493.48 | 1492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1489.30 | 1496.18 | 1494.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1489.30 | 1496.18 | 1494.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1492.00 | 1495.35 | 1494.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:30:00 | 1496.80 | 1500.52 | 1496.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 14:15:00 | 1492.00 | 1508.91 | 1509.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 1478.20 | 1493.85 | 1499.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1492.90 | 1492.09 | 1497.37 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 15:15:00 | 1509.80 | 1498.70 | 1498.35 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1491.70 | 1497.30 | 1497.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1477.40 | 1493.32 | 1495.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 1471.20 | 1466.83 | 1476.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 1471.20 | 1466.83 | 1476.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 1469.10 | 1467.28 | 1475.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 1469.10 | 1467.28 | 1475.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1447.20 | 1462.72 | 1472.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 1440.10 | 1460.58 | 1470.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:45:00 | 1440.80 | 1454.87 | 1461.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:15:00 | 1441.80 | 1450.94 | 1458.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:15:00 | 1439.50 | 1449.93 | 1455.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1448.00 | 1447.87 | 1453.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1451.70 | 1447.87 | 1453.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1442.10 | 1446.72 | 1452.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:45:00 | 1435.50 | 1444.82 | 1450.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1444.40 | 1430.40 | 1429.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1481.00 | 1449.39 | 1440.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 1482.10 | 1484.43 | 1466.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:00:00 | 1482.10 | 1484.43 | 1466.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1481.00 | 1483.24 | 1469.06 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1463.80 | 1467.90 | 1468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 10:15:00 | 1451.00 | 1464.47 | 1466.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 1462.30 | 1462.05 | 1464.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:45:00 | 1462.90 | 1462.05 | 1464.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1468.10 | 1463.26 | 1465.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 1466.90 | 1463.26 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1462.30 | 1463.07 | 1464.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1471.60 | 1463.07 | 1464.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1456.80 | 1461.82 | 1464.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 10:45:00 | 1454.20 | 1460.63 | 1463.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 1455.00 | 1452.58 | 1453.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 12:15:00 | 1473.80 | 1456.82 | 1455.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 12:15:00 | 1473.80 | 1456.82 | 1455.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 1483.00 | 1462.06 | 1457.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1469.60 | 1470.30 | 1463.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 1469.60 | 1470.30 | 1463.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1471.30 | 1477.27 | 1471.47 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1454.90 | 1467.32 | 1468.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1446.10 | 1463.08 | 1466.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1452.40 | 1452.30 | 1459.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 1452.40 | 1452.30 | 1459.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1455.10 | 1452.28 | 1457.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 1460.80 | 1452.28 | 1457.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1455.80 | 1452.98 | 1457.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 1456.60 | 1452.98 | 1457.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1467.70 | 1455.93 | 1458.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 1470.10 | 1455.93 | 1458.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1466.30 | 1458.00 | 1459.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:30:00 | 1467.80 | 1458.00 | 1459.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 1470.00 | 1461.31 | 1460.49 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 11:15:00 | 1457.00 | 1459.89 | 1460.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 12:15:00 | 1452.00 | 1458.31 | 1459.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1467.00 | 1439.86 | 1445.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1467.00 | 1439.86 | 1445.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1466.20 | 1445.13 | 1447.21 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1463.00 | 1448.70 | 1448.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 1478.30 | 1460.16 | 1454.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 11:15:00 | 1458.00 | 1459.73 | 1455.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:00:00 | 1458.00 | 1459.73 | 1455.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1450.00 | 1457.78 | 1454.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:45:00 | 1449.90 | 1457.78 | 1454.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1447.50 | 1455.73 | 1454.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 1448.30 | 1455.73 | 1454.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 1440.00 | 1452.58 | 1452.75 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 1456.00 | 1452.04 | 1451.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 15:15:00 | 1464.90 | 1455.25 | 1453.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1450.20 | 1457.89 | 1455.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1450.20 | 1457.89 | 1455.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1442.00 | 1454.71 | 1454.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 1445.50 | 1454.71 | 1454.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1443.30 | 1452.43 | 1453.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1350.60 | 1429.54 | 1442.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1300.80 | 1298.32 | 1336.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:00:00 | 1300.80 | 1298.32 | 1336.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1243.10 | 1240.02 | 1248.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1224.90 | 1235.73 | 1241.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 1214.70 | 1226.89 | 1234.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:00:00 | 1222.00 | 1212.47 | 1213.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 1218.90 | 1213.76 | 1213.74 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 13:15:00 | 1193.90 | 1209.79 | 1211.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1187.30 | 1200.28 | 1206.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 13:15:00 | 1196.40 | 1195.52 | 1202.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:00:00 | 1196.40 | 1195.52 | 1202.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1198.10 | 1195.95 | 1199.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:45:00 | 1199.90 | 1195.95 | 1199.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1197.50 | 1196.26 | 1199.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:15:00 | 1199.00 | 1196.26 | 1199.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1196.20 | 1196.25 | 1199.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 1187.40 | 1192.70 | 1197.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:00:00 | 1180.90 | 1168.37 | 1175.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 13:15:00 | 1213.50 | 1185.54 | 1182.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 13:15:00 | 1213.50 | 1185.54 | 1182.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 14:15:00 | 1232.10 | 1194.86 | 1186.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1195.60 | 1203.96 | 1194.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 1194.00 | 1203.96 | 1194.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1194.00 | 1201.97 | 1194.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:15:00 | 1193.10 | 1201.97 | 1194.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1193.60 | 1200.30 | 1194.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:15:00 | 1191.40 | 1200.30 | 1194.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1173.90 | 1193.93 | 1192.65 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 10:15:00 | 1181.00 | 1189.98 | 1190.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 1179.60 | 1187.90 | 1189.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1182.60 | 1177.27 | 1183.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1182.60 | 1177.27 | 1183.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1189.90 | 1179.80 | 1183.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 1189.90 | 1179.80 | 1183.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 1191.20 | 1182.08 | 1184.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 12:00:00 | 1191.20 | 1182.08 | 1184.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 1189.60 | 1185.79 | 1185.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 1195.00 | 1188.29 | 1186.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1181.90 | 1190.60 | 1188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1181.90 | 1190.60 | 1188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1179.90 | 1188.46 | 1188.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1179.90 | 1188.46 | 1188.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 1180.00 | 1186.77 | 1187.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1139.40 | 1177.29 | 1182.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1156.60 | 1153.56 | 1163.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 13:00:00 | 1156.60 | 1153.56 | 1163.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1144.00 | 1151.39 | 1159.47 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 1170.00 | 1163.10 | 1162.85 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1154.60 | 1162.10 | 1162.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1145.80 | 1155.50 | 1159.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1153.30 | 1145.36 | 1150.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:30:00 | 1154.20 | 1145.36 | 1150.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1155.50 | 1147.39 | 1150.53 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 1174.90 | 1154.45 | 1153.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 13:15:00 | 1192.80 | 1179.07 | 1170.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1176.90 | 1181.17 | 1173.76 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 1167.50 | 1171.00 | 1171.33 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1177.00 | 1170.97 | 1170.94 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 1166.50 | 1170.07 | 1170.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 10:15:00 | 1164.70 | 1169.00 | 1170.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1181.30 | 1166.50 | 1167.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1181.30 | 1166.50 | 1167.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1189.20 | 1171.04 | 1169.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 11:15:00 | 1190.30 | 1174.89 | 1171.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 15:15:00 | 1184.20 | 1184.62 | 1178.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1172.90 | 1184.62 | 1178.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1174.80 | 1182.66 | 1177.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 1171.20 | 1182.66 | 1177.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1170.00 | 1180.13 | 1177.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 1169.40 | 1180.13 | 1177.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 1182.80 | 1180.18 | 1177.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 1185.90 | 1180.18 | 1177.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 1185.80 | 1184.11 | 1180.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 12:15:00 | 1174.00 | 1182.00 | 1180.65 | SL hit (close<static) qty=1.00 sl=1175.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1174.60 | 1179.42 | 1179.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 15:15:00 | 1167.00 | 1176.93 | 1178.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1169.90 | 1164.50 | 1168.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 1172.00 | 1164.50 | 1168.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1159.50 | 1163.50 | 1168.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 15:15:00 | 1151.70 | 1163.25 | 1164.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 1151.50 | 1159.57 | 1162.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1177.50 | 1163.37 | 1163.59 | SL hit (close>static) qty=1.00 sl=1172.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 1172.90 | 1165.28 | 1164.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1195.70 | 1174.00 | 1168.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1188.50 | 1193.21 | 1185.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 13:00:00 | 1188.50 | 1193.21 | 1185.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1192.20 | 1193.01 | 1186.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 1194.00 | 1193.01 | 1186.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 1193.70 | 1192.46 | 1187.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1178.00 | 1186.72 | 1186.47 | SL hit (close<static) qty=1.00 sl=1182.20 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 1178.00 | 1184.98 | 1185.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 09:15:00 | 1156.30 | 1179.24 | 1183.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1153.80 | 1140.66 | 1150.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1153.80 | 1140.66 | 1150.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1147.00 | 1141.93 | 1150.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:45:00 | 1151.20 | 1141.93 | 1150.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1102.60 | 1096.62 | 1106.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1100.40 | 1096.62 | 1106.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1101.20 | 1097.54 | 1106.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1098.90 | 1098.21 | 1105.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:45:00 | 1098.50 | 1098.65 | 1105.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 1094.40 | 1098.65 | 1105.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1092.70 | 1094.42 | 1099.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1093.60 | 1094.25 | 1099.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1099.30 | 1094.25 | 1099.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1097.80 | 1094.57 | 1098.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:30:00 | 1096.90 | 1094.57 | 1098.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1094.90 | 1094.64 | 1098.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1092.90 | 1096.01 | 1097.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 1092.70 | 1096.01 | 1097.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 1088.80 | 1094.57 | 1097.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1043.95 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1043.58 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 1039.68 | 1051.29 | 1059.52 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.07 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.26 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 12:15:00 | 1038.07 | 1048.83 | 1057.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1054.70 | 1045.75 | 1052.90 | SL hit (close>ema200) qty=0.50 sl=1045.75 alert=retest2 |

### Cycle 170 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 1068.60 | 1057.15 | 1056.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 14:15:00 | 1076.80 | 1061.08 | 1058.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1066.90 | 1069.61 | 1064.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 1066.90 | 1069.61 | 1064.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1065.50 | 1068.79 | 1064.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1065.50 | 1068.79 | 1064.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1065.00 | 1068.03 | 1064.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 15:15:00 | 1064.00 | 1068.03 | 1064.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1064.00 | 1067.23 | 1064.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1059.10 | 1067.23 | 1064.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1053.00 | 1064.38 | 1063.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 1053.00 | 1064.38 | 1063.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1043.50 | 1060.20 | 1061.74 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 1056.40 | 1051.99 | 1051.85 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1048.20 | 1051.72 | 1051.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1045.00 | 1049.93 | 1051.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1049.60 | 1049.06 | 1050.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1050.70 | 1049.06 | 1050.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1053.60 | 1049.96 | 1050.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 1044.60 | 1049.96 | 1050.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1040.70 | 1048.11 | 1049.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 1039.70 | 1046.17 | 1048.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:15:00 | 1039.70 | 1044.03 | 1047.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:00:00 | 1038.60 | 1042.95 | 1046.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 1039.00 | 1041.42 | 1045.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1049.60 | 1042.11 | 1045.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 1049.60 | 1042.11 | 1045.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1053.50 | 1044.39 | 1045.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 1052.80 | 1044.39 | 1045.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1056.90 | 1048.68 | 1047.60 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 1046.50 | 1053.10 | 1053.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 1044.20 | 1051.32 | 1052.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1047.20 | 1046.89 | 1049.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 1045.20 | 1046.89 | 1049.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1047.00 | 1046.91 | 1049.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1050.50 | 1046.91 | 1049.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1047.90 | 1047.11 | 1049.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:45:00 | 1048.00 | 1047.11 | 1049.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1049.90 | 1047.67 | 1049.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 1046.70 | 1047.67 | 1049.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1052.00 | 1047.75 | 1048.79 | SL hit (close>static) qty=1.00 sl=1050.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 11:15:00 | 1051.50 | 1049.01 | 1048.87 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 12:15:00 | 1029.10 | 1045.03 | 1047.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 13:15:00 | 1025.00 | 1041.02 | 1045.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 11:15:00 | 1010.90 | 1010.48 | 1020.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 12:00:00 | 1010.90 | 1010.48 | 1020.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1025.30 | 1013.44 | 1021.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 13:00:00 | 1025.30 | 1013.44 | 1021.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1021.00 | 1014.95 | 1021.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 1007.60 | 1013.32 | 1018.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 957.22 | 976.56 | 989.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 950.80 | 949.45 | 966.60 | SL hit (close>ema200) qty=0.50 sl=949.45 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 951.00 | 944.21 | 943.51 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 931.50 | 941.20 | 942.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 925.90 | 936.64 | 940.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 933.40 | 929.01 | 934.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 12:30:00 | 923.50 | 928.20 | 930.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 952.90 | 932.92 | 931.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 952.90 | 932.92 | 931.28 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 926.70 | 929.77 | 930.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 918.30 | 927.48 | 929.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 927.70 | 926.65 | 928.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 927.70 | 926.65 | 928.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 924.10 | 926.14 | 927.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 920.10 | 925.11 | 927.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 954.20 | 924.76 | 924.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 954.20 | 924.76 | 924.52 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 09:15:00 | 920.10 | 926.23 | 926.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 09:15:00 | 916.10 | 920.27 | 922.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 12:15:00 | 919.30 | 918.43 | 921.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-27 13:00:00 | 919.30 | 918.43 | 921.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 921.30 | 919.00 | 921.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 916.60 | 918.81 | 920.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 915.00 | 918.15 | 920.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 907.55 | 893.12 | 891.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 907.55 | 893.12 | 891.63 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 882.95 | 891.13 | 891.69 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 896.10 | 892.12 | 892.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 900.00 | 895.48 | 893.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 895.20 | 895.42 | 894.01 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:15:00 | 906.15 | 895.42 | 894.01 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 908.00 | 910.32 | 906.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 908.45 | 910.32 | 906.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 909.70 | 910.19 | 907.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 904.60 | 908.70 | 907.19 | SL hit (close<ema400) qty=1.00 sl=907.19 alert=retest1 |

### Cycle 187 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 898.10 | 904.77 | 905.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 891.15 | 900.62 | 903.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 905.70 | 889.53 | 893.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 905.70 | 889.53 | 893.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 890.10 | 889.64 | 892.90 | EMA400 retest candle locked (from downside) |

### Cycle 188 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 902.30 | 895.96 | 895.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 910.15 | 899.52 | 896.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 899.65 | 904.23 | 901.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 898.05 | 904.23 | 901.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 912.25 | 905.84 | 902.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 899.00 | 905.84 | 902.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 901.35 | 905.63 | 903.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:15:00 | 903.60 | 905.63 | 903.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 900.55 | 904.61 | 903.41 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 898.15 | 902.62 | 902.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 897.45 | 901.58 | 902.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 879.85 | 876.51 | 884.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 879.85 | 876.51 | 884.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 869.45 | 875.10 | 883.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 884.60 | 875.10 | 883.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 892.95 | 877.87 | 882.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 892.95 | 877.87 | 882.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 891.15 | 880.53 | 883.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 15:00:00 | 884.55 | 883.42 | 884.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 869.20 | 866.55 | 866.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 869.20 | 866.55 | 866.42 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 858.95 | 865.51 | 866.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 854.45 | 860.25 | 863.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 855.00 | 853.40 | 856.51 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 861.05 | 857.64 | 857.19 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 855.10 | 858.95 | 859.01 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 10:15:00 | 858.95 | 856.96 | 856.95 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 11:15:00 | 856.80 | 856.93 | 856.94 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 862.95 | 857.91 | 857.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 11:15:00 | 873.80 | 862.58 | 859.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 867.35 | 870.55 | 866.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 10:30:00 | 869.15 | 870.55 | 866.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 853.90 | 867.22 | 864.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 853.90 | 867.22 | 864.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 853.25 | 864.43 | 863.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 853.25 | 864.43 | 863.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 854.95 | 862.53 | 863.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 11:15:00 | 850.10 | 855.90 | 859.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 856.35 | 855.10 | 858.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:45:00 | 856.00 | 855.10 | 858.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 857.00 | 855.54 | 857.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 855.05 | 855.54 | 857.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 856.85 | 855.80 | 857.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 852.00 | 855.99 | 857.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 861.00 | 851.53 | 852.04 | SL hit (close>static) qty=1.00 sl=858.95 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 861.00 | 853.43 | 852.85 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 831.50 | 851.64 | 852.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 807.80 | 836.91 | 845.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 825.00 | 804.91 | 815.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 820.45 | 808.83 | 816.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:30:00 | 822.05 | 814.57 | 817.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 15:00:00 | 822.00 | 817.72 | 818.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 816.10 | 818.72 | 819.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 809.75 | 808.70 | 811.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:30:00 | 809.70 | 808.70 | 811.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 800.60 | 800.32 | 804.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:30:00 | 799.80 | 799.99 | 803.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 799.35 | 800.01 | 803.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 14:30:00 | 799.55 | 799.84 | 802.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 816.75 | 803.50 | 803.80 | SL hit (close>static) qty=1.00 sl=805.20 alert=retest2 |

### Cycle 200 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 810.05 | 804.81 | 804.37 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 799.50 | 803.68 | 804.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 15:15:00 | 796.00 | 802.15 | 803.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 763.50 | 763.24 | 773.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:30:00 | 763.65 | 763.24 | 773.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 744.30 | 746.41 | 751.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 741.75 | 746.41 | 751.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 13:15:00 | 762.20 | 750.48 | 751.97 | SL hit (close>static) qty=1.00 sl=755.25 alert=retest2 |

### Cycle 202 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 761.00 | 754.01 | 753.40 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 746.45 | 752.11 | 752.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 744.10 | 750.51 | 751.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 722.05 | 719.87 | 729.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 10:30:00 | 721.25 | 719.87 | 729.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 725.60 | 721.28 | 728.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 727.85 | 721.28 | 728.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 725.85 | 722.19 | 728.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:30:00 | 728.75 | 722.19 | 728.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 729.00 | 723.55 | 728.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 729.00 | 723.55 | 728.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 728.70 | 724.58 | 728.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 718.05 | 724.58 | 728.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:45:00 | 726.40 | 724.42 | 727.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:00:00 | 726.20 | 725.02 | 726.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 731.05 | 724.26 | 725.27 | SL hit (close>static) qty=1.00 sl=729.40 alert=retest2 |

### Cycle 204 — BUY (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 12:15:00 | 729.85 | 724.42 | 724.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 13:15:00 | 735.60 | 726.65 | 725.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 724.30 | 728.46 | 726.67 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 716.00 | 724.82 | 725.25 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 742.00 | 727.54 | 726.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 754.10 | 740.30 | 734.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 761.80 | 763.29 | 752.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 11:00:00 | 761.80 | 763.29 | 752.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 752.80 | 759.06 | 754.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 734.30 | 759.06 | 754.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 736.50 | 754.55 | 752.62 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 737.40 | 751.12 | 751.23 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 751.80 | 747.54 | 747.39 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 742.80 | 747.51 | 747.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 727.05 | 742.14 | 745.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 740.70 | 738.65 | 742.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 14:00:00 | 740.70 | 738.65 | 742.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 702.30 | 694.79 | 700.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 704.95 | 694.79 | 700.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 709.95 | 697.82 | 701.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 710.60 | 697.82 | 701.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 705.70 | 699.40 | 701.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:15:00 | 711.60 | 699.40 | 701.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 716.50 | 702.82 | 703.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 716.50 | 702.82 | 703.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 718.85 | 706.02 | 704.71 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 696.25 | 704.48 | 705.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 687.35 | 697.58 | 700.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 684.35 | 683.61 | 690.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 684.35 | 683.61 | 690.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 690.50 | 684.99 | 690.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 692.90 | 684.99 | 690.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 704.15 | 688.82 | 691.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 703.55 | 688.82 | 691.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 707.75 | 692.61 | 692.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 707.75 | 692.61 | 692.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 212 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 699.65 | 694.01 | 693.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 720.85 | 699.65 | 696.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 702.85 | 707.60 | 702.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 702.85 | 707.60 | 702.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 703.00 | 706.68 | 702.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 701.00 | 706.68 | 702.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 696.50 | 704.65 | 701.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 693.65 | 704.65 | 701.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 690.15 | 701.75 | 700.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 691.40 | 701.75 | 700.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 682.75 | 697.95 | 699.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 680.90 | 690.67 | 695.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 695.35 | 673.31 | 680.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 695.35 | 673.31 | 680.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 700.05 | 678.66 | 682.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 700.05 | 678.66 | 682.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 709.95 | 684.92 | 684.82 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 690.75 | 694.81 | 695.21 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 716.05 | 698.10 | 696.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 722.25 | 702.93 | 698.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 720.45 | 720.67 | 711.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 720.45 | 720.67 | 711.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 716.50 | 721.21 | 714.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 716.90 | 721.21 | 714.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 717.95 | 720.56 | 714.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 715.75 | 720.56 | 714.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 722.30 | 724.20 | 720.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 726.50 | 724.20 | 720.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:30:00 | 725.95 | 723.20 | 721.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 733.50 | 722.91 | 721.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 740.00 | 747.89 | 748.61 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 755.85 | 750.30 | 749.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 766.45 | 755.66 | 752.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 12:15:00 | 802.55 | 802.90 | 790.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 12:30:00 | 803.00 | 802.90 | 790.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 807.00 | 810.46 | 803.37 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 799.00 | 801.41 | 801.56 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 811.45 | 803.42 | 802.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 818.65 | 810.85 | 806.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 809.55 | 813.18 | 808.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 809.55 | 813.18 | 808.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 806.30 | 811.80 | 808.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 803.85 | 811.80 | 808.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 812.10 | 811.86 | 808.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:15:00 | 814.80 | 811.86 | 808.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 11:00:00 | 813.70 | 819.41 | 817.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 13:15:00 | 896.28 | 882.25 | 866.44 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-15 13:30:00 | 1520.00 | 2023-05-18 14:15:00 | 1514.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-05-17 12:15:00 | 1512.30 | 2023-05-18 14:15:00 | 1514.80 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2023-05-17 13:00:00 | 1513.15 | 2023-05-18 14:15:00 | 1514.80 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2023-05-17 15:00:00 | 1528.00 | 2023-05-18 14:15:00 | 1514.80 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-05-22 09:15:00 | 1483.80 | 2023-05-29 09:15:00 | 1409.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-23 11:00:00 | 1494.15 | 2023-05-29 09:15:00 | 1419.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-23 12:30:00 | 1489.30 | 2023-05-29 09:15:00 | 1414.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-22 09:15:00 | 1483.80 | 2023-05-29 13:15:00 | 1428.00 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2023-05-23 11:00:00 | 1494.15 | 2023-05-29 13:15:00 | 1428.00 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2023-05-23 12:30:00 | 1489.30 | 2023-05-29 13:15:00 | 1428.00 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2023-06-08 11:15:00 | 1403.00 | 2023-06-08 13:15:00 | 1415.35 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2023-06-08 15:15:00 | 1403.00 | 2023-06-13 14:15:00 | 1332.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-08 15:15:00 | 1403.00 | 2023-06-14 10:15:00 | 1355.10 | STOP_HIT | 0.50 | 3.41% |
| SELL | retest2 | 2023-06-09 09:30:00 | 1398.60 | 2023-06-15 09:15:00 | 1384.00 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2023-06-22 11:30:00 | 1363.00 | 2023-06-26 15:15:00 | 1378.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-06-23 11:00:00 | 1361.00 | 2023-06-26 15:15:00 | 1378.50 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2023-06-23 12:15:00 | 1364.80 | 2023-06-26 15:15:00 | 1378.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2023-06-23 12:45:00 | 1364.95 | 2023-06-26 15:15:00 | 1378.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-07-11 12:15:00 | 1348.20 | 2023-07-17 13:15:00 | 1318.00 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2023-07-24 14:30:00 | 1313.45 | 2023-07-31 09:15:00 | 1330.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2023-08-16 09:15:00 | 1417.30 | 2023-08-16 11:15:00 | 1405.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2023-08-21 13:30:00 | 1371.85 | 2023-08-22 09:15:00 | 1399.65 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-08-21 14:15:00 | 1373.15 | 2023-08-22 09:15:00 | 1399.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2023-08-24 09:30:00 | 1409.30 | 2023-08-25 11:15:00 | 1394.95 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-08-24 12:30:00 | 1408.20 | 2023-08-25 11:15:00 | 1394.95 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2023-08-24 14:15:00 | 1405.65 | 2023-08-25 11:15:00 | 1394.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2023-08-25 09:15:00 | 1409.00 | 2023-08-25 11:15:00 | 1394.95 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-09-11 09:15:00 | 1495.05 | 2023-09-12 11:15:00 | 1453.25 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2023-09-12 10:30:00 | 1467.05 | 2023-09-12 11:15:00 | 1453.25 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-09-21 12:30:00 | 1431.50 | 2023-09-27 14:15:00 | 1426.55 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2023-09-22 09:45:00 | 1421.15 | 2023-09-27 14:15:00 | 1426.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2023-09-29 15:00:00 | 1406.90 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-10-03 09:30:00 | 1409.90 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2023-10-03 12:15:00 | 1410.15 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-10-03 13:00:00 | 1409.05 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-10-04 11:15:00 | 1408.65 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2023-10-04 14:15:00 | 1410.50 | 2023-10-05 09:15:00 | 1417.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2023-10-10 14:15:00 | 1379.65 | 2023-10-11 11:15:00 | 1391.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2023-10-19 09:15:00 | 1368.00 | 2023-10-26 09:15:00 | 1299.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 13:45:00 | 1371.10 | 2023-10-26 09:15:00 | 1302.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 14:30:00 | 1371.15 | 2023-10-26 09:15:00 | 1302.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:15:00 | 1367.85 | 2023-10-26 09:15:00 | 1299.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 11:15:00 | 1364.30 | 2023-10-26 09:15:00 | 1296.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:30:00 | 1362.05 | 2023-10-26 09:15:00 | 1293.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 09:15:00 | 1368.00 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2023-10-19 13:45:00 | 1371.10 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2023-10-19 14:30:00 | 1371.15 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.73% |
| SELL | retest2 | 2023-10-20 09:15:00 | 1367.85 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2023-10-20 11:15:00 | 1364.30 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.25% |
| SELL | retest2 | 2023-10-20 12:30:00 | 1362.05 | 2023-10-26 15:15:00 | 1320.00 | STOP_HIT | 0.50 | 3.09% |
| BUY | retest2 | 2023-11-06 09:15:00 | 1345.20 | 2023-11-16 09:15:00 | 1367.75 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2023-11-06 09:45:00 | 1345.00 | 2023-11-16 09:15:00 | 1367.75 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2023-11-22 12:15:00 | 1340.00 | 2023-11-23 09:15:00 | 1357.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2023-11-28 10:15:00 | 1345.00 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2023-11-29 11:45:00 | 1345.00 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.77% |
| SELL | retest2 | 2023-11-29 12:45:00 | 1342.75 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2023-11-29 13:30:00 | 1343.85 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2023-12-05 09:15:00 | 1450.90 | 2023-12-05 12:15:00 | 1414.00 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest1 | 2023-12-05 10:00:00 | 1439.75 | 2023-12-05 12:15:00 | 1414.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-12-18 09:15:00 | 1500.00 | 2023-12-20 13:15:00 | 1459.90 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2023-12-28 11:15:00 | 1541.90 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2023-12-28 12:15:00 | 1545.00 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2023-12-29 09:15:00 | 1550.95 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2023-12-29 10:15:00 | 1542.70 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.77% |
| BUY | retest2 | 2023-12-29 11:30:00 | 1552.80 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2023-12-29 12:15:00 | 1553.55 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2023-12-29 13:00:00 | 1554.85 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2023-12-29 15:15:00 | 1554.00 | 2024-01-03 15:15:00 | 1570.00 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2024-01-05 10:45:00 | 1566.60 | 2024-01-10 09:15:00 | 1491.21 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2024-01-05 12:15:00 | 1569.70 | 2024-01-10 12:15:00 | 1488.27 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2024-01-05 14:00:00 | 1567.25 | 2024-01-10 12:15:00 | 1488.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-08 09:15:00 | 1567.00 | 2024-01-10 12:15:00 | 1488.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-05 10:45:00 | 1566.60 | 2024-01-11 09:15:00 | 1506.05 | STOP_HIT | 0.50 | 3.87% |
| SELL | retest2 | 2024-01-05 12:15:00 | 1569.70 | 2024-01-11 09:15:00 | 1506.05 | STOP_HIT | 0.50 | 4.05% |
| SELL | retest2 | 2024-01-05 14:00:00 | 1567.25 | 2024-01-11 09:15:00 | 1506.05 | STOP_HIT | 0.50 | 3.90% |
| SELL | retest2 | 2024-01-08 09:15:00 | 1567.00 | 2024-01-11 09:15:00 | 1506.05 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest2 | 2024-01-08 10:45:00 | 1524.95 | 2024-01-12 09:15:00 | 1529.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-17 09:15:00 | 1492.85 | 2024-01-19 09:15:00 | 1497.05 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-01-19 10:45:00 | 1491.00 | 2024-01-19 14:15:00 | 1502.70 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-01-19 13:45:00 | 1490.40 | 2024-01-19 14:15:00 | 1502.70 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-01-29 10:30:00 | 1431.80 | 2024-01-31 09:15:00 | 1458.10 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-02-08 09:30:00 | 1396.70 | 2024-02-15 09:15:00 | 1401.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-02-08 10:30:00 | 1397.00 | 2024-02-15 09:15:00 | 1401.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-02-12 10:00:00 | 1400.00 | 2024-02-15 11:15:00 | 1399.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-02-13 10:45:00 | 1394.20 | 2024-02-15 11:15:00 | 1399.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2024-02-14 09:15:00 | 1390.00 | 2024-02-15 11:15:00 | 1399.00 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-02-14 09:45:00 | 1388.10 | 2024-02-15 11:15:00 | 1399.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-02-20 09:15:00 | 1408.00 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-02-20 09:45:00 | 1405.65 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2024-02-20 10:15:00 | 1406.20 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | 1.84% |
| BUY | retest2 | 2024-02-20 12:15:00 | 1405.30 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2024-02-26 11:15:00 | 1457.75 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-02-26 14:30:00 | 1467.00 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-02-27 10:30:00 | 1456.95 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-02-27 12:30:00 | 1457.60 | 2024-02-28 10:15:00 | 1432.05 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-03-04 10:45:00 | 1398.00 | 2024-03-12 10:15:00 | 1328.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 12:00:00 | 1398.70 | 2024-03-12 10:15:00 | 1328.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 12:30:00 | 1398.05 | 2024-03-12 10:15:00 | 1328.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 09:15:00 | 1393.20 | 2024-03-12 11:15:00 | 1323.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 10:45:00 | 1398.00 | 2024-03-12 14:15:00 | 1335.95 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2024-03-04 12:00:00 | 1398.70 | 2024-03-12 14:15:00 | 1335.95 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2024-03-04 12:30:00 | 1398.05 | 2024-03-12 14:15:00 | 1335.95 | STOP_HIT | 0.50 | 4.44% |
| SELL | retest2 | 2024-03-06 09:15:00 | 1393.20 | 2024-03-12 14:15:00 | 1335.95 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2024-03-06 11:00:00 | 1371.00 | 2024-03-13 10:15:00 | 1302.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 09:30:00 | 1372.00 | 2024-03-13 10:15:00 | 1303.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-06 11:00:00 | 1371.00 | 2024-03-14 09:15:00 | 1310.50 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2024-03-11 09:30:00 | 1372.00 | 2024-03-14 09:15:00 | 1310.50 | STOP_HIT | 0.50 | 4.48% |
| BUY | retest2 | 2024-03-28 14:15:00 | 1343.20 | 2024-04-04 14:15:00 | 1335.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-03-28 15:00:00 | 1341.80 | 2024-04-04 14:15:00 | 1335.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-04-01 09:15:00 | 1349.95 | 2024-04-04 14:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-04-02 14:00:00 | 1340.15 | 2024-04-04 14:15:00 | 1335.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-04-18 09:15:00 | 1332.05 | 2024-04-25 12:15:00 | 1314.50 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2024-04-18 12:15:00 | 1331.85 | 2024-04-25 12:15:00 | 1314.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2024-04-29 09:15:00 | 1319.05 | 2024-05-02 10:15:00 | 1313.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-04-29 10:45:00 | 1318.95 | 2024-05-02 10:15:00 | 1313.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-04-29 11:15:00 | 1321.80 | 2024-05-02 10:15:00 | 1313.50 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-04-29 11:45:00 | 1320.95 | 2024-05-02 11:15:00 | 1309.15 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-30 09:15:00 | 1328.55 | 2024-05-02 11:15:00 | 1309.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-04-30 12:15:00 | 1327.40 | 2024-05-02 11:15:00 | 1309.15 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2024-04-30 13:30:00 | 1327.35 | 2024-05-02 11:15:00 | 1309.15 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-05-07 09:15:00 | 1302.45 | 2024-05-15 09:15:00 | 1304.90 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-05-22 09:15:00 | 1346.60 | 2024-05-27 09:15:00 | 1354.95 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-06-03 11:15:00 | 1306.00 | 2024-06-06 11:15:00 | 1313.90 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-06-03 14:45:00 | 1305.85 | 2024-06-06 11:15:00 | 1313.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-06-04 10:30:00 | 1298.05 | 2024-06-06 11:15:00 | 1313.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-06-05 11:30:00 | 1302.60 | 2024-06-06 11:15:00 | 1313.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-06-25 12:15:00 | 1457.00 | 2024-06-26 13:15:00 | 1438.05 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-25 12:45:00 | 1451.90 | 2024-06-26 13:15:00 | 1438.05 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-06-26 10:45:00 | 1451.70 | 2024-06-26 13:15:00 | 1438.05 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-27 13:45:00 | 1425.60 | 2024-07-01 09:15:00 | 1481.50 | STOP_HIT | 1.00 | -3.92% |
| BUY | retest2 | 2024-07-08 13:15:00 | 1510.05 | 2024-07-09 12:15:00 | 1485.25 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1459.10 | 2024-07-24 10:15:00 | 1460.85 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-07-22 13:30:00 | 1474.10 | 2024-07-24 10:15:00 | 1460.85 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2024-07-23 15:00:00 | 1457.05 | 2024-07-24 10:15:00 | 1460.85 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-07-30 09:15:00 | 1525.10 | 2024-08-01 09:15:00 | 1677.61 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-08-06 10:30:00 | 1594.10 | 2024-08-08 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-08-07 10:30:00 | 1598.95 | 2024-08-08 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-08-07 11:30:00 | 1598.80 | 2024-08-08 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-08-07 12:30:00 | 1599.15 | 2024-08-08 09:15:00 | 1625.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-08-14 12:30:00 | 1592.55 | 2024-08-26 10:15:00 | 1514.87 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2024-08-14 13:45:00 | 1594.60 | 2024-08-26 13:15:00 | 1512.92 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2024-08-14 12:30:00 | 1592.55 | 2024-08-27 09:15:00 | 1545.00 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2024-08-14 13:45:00 | 1594.60 | 2024-08-27 09:15:00 | 1545.00 | STOP_HIT | 0.50 | 3.11% |
| SELL | retest2 | 2024-08-22 10:00:00 | 1585.00 | 2024-08-29 10:15:00 | 1505.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-22 10:00:00 | 1585.00 | 2024-08-30 13:15:00 | 1487.45 | STOP_HIT | 0.50 | 6.15% |
| BUY | retest2 | 2024-09-12 13:45:00 | 1574.40 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-12 15:00:00 | 1574.40 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-13 09:45:00 | 1575.55 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-09-13 14:45:00 | 1576.50 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-09-17 12:45:00 | 1579.90 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-17 14:45:00 | 1580.00 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-09-18 10:15:00 | 1580.70 | 2024-09-18 12:15:00 | 1557.10 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-09-20 11:15:00 | 1551.80 | 2024-09-26 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-09-20 12:45:00 | 1559.55 | 2024-09-26 14:15:00 | 1559.95 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-10-14 11:45:00 | 1584.10 | 2024-10-18 12:15:00 | 1595.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1573.80 | 2024-10-23 09:15:00 | 1495.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:15:00 | 1573.80 | 2024-10-24 11:15:00 | 1529.40 | STOP_HIT | 0.50 | 2.82% |
| BUY | retest2 | 2024-11-05 13:15:00 | 1511.80 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1525.50 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-11-25 09:45:00 | 1291.35 | 2024-11-25 10:15:00 | 1299.85 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-11-25 12:45:00 | 1292.80 | 2024-11-26 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-11-25 13:30:00 | 1290.60 | 2024-11-26 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-11-25 15:00:00 | 1287.95 | 2024-11-26 10:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-12-09 09:15:00 | 1298.40 | 2024-12-09 12:15:00 | 1288.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-12-13 11:45:00 | 1423.85 | 2024-12-23 09:15:00 | 1424.55 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-12-16 12:15:00 | 1421.00 | 2024-12-23 09:15:00 | 1424.55 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-12-17 10:45:00 | 1434.80 | 2024-12-23 09:15:00 | 1424.55 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1407.80 | 2024-12-31 10:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-12-30 12:00:00 | 1414.00 | 2024-12-31 10:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-12-30 13:00:00 | 1414.95 | 2024-12-31 10:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-12-31 10:00:00 | 1416.25 | 2024-12-31 10:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-01-09 12:45:00 | 1449.30 | 2025-01-13 13:15:00 | 1376.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 14:45:00 | 1448.50 | 2025-01-13 13:15:00 | 1376.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:45:00 | 1449.30 | 2025-01-15 09:15:00 | 1377.75 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-01-09 14:45:00 | 1448.50 | 2025-01-15 09:15:00 | 1377.75 | STOP_HIT | 0.50 | 4.88% |
| SELL | retest2 | 2025-01-30 13:45:00 | 1349.05 | 2025-01-30 14:15:00 | 1364.80 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-02-01 14:45:00 | 1424.00 | 2025-02-06 15:15:00 | 1463.00 | STOP_HIT | 1.00 | 2.74% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1439.35 | 2025-02-11 11:15:00 | 1367.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1439.35 | 2025-02-14 10:15:00 | 1295.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1279.30 | 2025-02-28 09:15:00 | 1215.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1279.30 | 2025-03-03 09:15:00 | 1151.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-12 11:15:00 | 1202.45 | 2025-03-18 10:15:00 | 1216.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-03-24 09:15:00 | 1245.80 | 2025-03-24 10:15:00 | 1231.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-03-28 11:15:00 | 1190.70 | 2025-04-01 14:15:00 | 1217.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-03-28 11:45:00 | 1188.50 | 2025-04-01 14:15:00 | 1217.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-04-01 13:30:00 | 1191.00 | 2025-04-01 14:15:00 | 1217.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1161.10 | 2025-04-11 12:15:00 | 1179.50 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1158.30 | 2025-04-11 12:15:00 | 1179.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-04-09 13:45:00 | 1163.40 | 2025-04-11 12:15:00 | 1179.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-04-21 13:15:00 | 1205.00 | 2025-04-25 10:15:00 | 1181.30 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-04-22 09:30:00 | 1206.30 | 2025-04-25 10:15:00 | 1181.30 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-04-23 09:15:00 | 1205.60 | 2025-04-25 10:15:00 | 1181.30 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-04-23 12:45:00 | 1203.70 | 2025-04-25 10:15:00 | 1181.30 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-05-05 13:15:00 | 1178.20 | 2025-05-05 13:15:00 | 1187.20 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-05-07 13:30:00 | 1179.30 | 2025-05-08 09:15:00 | 1223.00 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-05-07 14:15:00 | 1178.50 | 2025-05-08 09:15:00 | 1223.00 | STOP_HIT | 1.00 | -3.78% |
| BUY | retest2 | 2025-05-30 15:00:00 | 1548.10 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -3.62% |
| BUY | retest2 | 2025-06-03 13:15:00 | 1491.70 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-06-03 14:00:00 | 1491.20 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1496.10 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-06-05 09:30:00 | 1496.80 | 2025-06-06 14:15:00 | 1492.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-16 10:30:00 | 1440.10 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-06-17 14:45:00 | 1440.80 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-06-18 10:15:00 | 1441.80 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-06-18 15:15:00 | 1439.50 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-19 11:45:00 | 1435.50 | 2025-06-24 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-01 10:45:00 | 1454.20 | 2025-07-03 12:15:00 | 1473.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-03 12:00:00 | 1455.00 | 2025-07-03 12:15:00 | 1473.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 13:45:00 | 1224.90 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-08-01 11:45:00 | 1214.70 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-08-06 12:00:00 | 1222.00 | 2025-08-06 12:15:00 | 1218.90 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-08-08 14:45:00 | 1187.40 | 2025-08-13 13:15:00 | 1213.50 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-08-13 10:00:00 | 1180.90 | 2025-08-13 13:15:00 | 1213.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-09-08 13:15:00 | 1185.90 | 2025-09-09 12:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1185.80 | 2025-09-09 12:15:00 | 1174.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-15 15:15:00 | 1151.70 | 2025-09-16 12:15:00 | 1177.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-16 09:30:00 | 1151.50 | 2025-09-16 12:15:00 | 1177.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-09-18 14:15:00 | 1194.00 | 2025-09-19 14:15:00 | 1178.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-09-19 10:15:00 | 1193.70 | 2025-09-19 14:15:00 | 1178.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1098.90 | 2025-10-09 11:15:00 | 1043.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 10:45:00 | 1098.50 | 2025-10-09 11:15:00 | 1043.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1094.40 | 2025-10-09 11:15:00 | 1039.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1092.70 | 2025-10-09 12:15:00 | 1038.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1092.90 | 2025-10-09 12:15:00 | 1038.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 11:15:00 | 1092.70 | 2025-10-09 12:15:00 | 1038.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1098.90 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-09-30 10:45:00 | 1098.50 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.99% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1094.40 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1092.70 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1092.90 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.50% |
| SELL | retest2 | 2025-10-03 11:15:00 | 1092.70 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.48% |
| SELL | retest2 | 2025-10-03 12:00:00 | 1088.80 | 2025-10-10 09:15:00 | 1034.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 12:00:00 | 1088.80 | 2025-10-10 09:15:00 | 1054.70 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-10-20 10:45:00 | 1039.70 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-10-20 13:15:00 | 1039.70 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-10-20 14:00:00 | 1038.60 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-20 14:45:00 | 1039.00 | 2025-10-23 10:15:00 | 1056.90 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-29 09:15:00 | 1046.70 | 2025-10-29 11:15:00 | 1052.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-10-29 12:45:00 | 1045.20 | 2025-10-30 11:15:00 | 1051.50 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-11-04 10:00:00 | 1007.60 | 2025-11-07 09:15:00 | 957.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 10:00:00 | 1007.60 | 2025-11-10 09:15:00 | 950.80 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2025-11-19 12:30:00 | 923.50 | 2025-11-20 11:15:00 | 952.90 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-11-21 13:45:00 | 920.10 | 2025-11-24 14:15:00 | 954.20 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-11-28 09:15:00 | 916.60 | 2025-12-10 09:15:00 | 907.55 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-11-28 09:45:00 | 915.00 | 2025-12-10 09:15:00 | 907.55 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest1 | 2025-12-12 09:15:00 | 906.15 | 2025-12-16 12:15:00 | 904.60 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-12-30 15:00:00 | 884.55 | 2026-01-07 11:15:00 | 869.20 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2026-01-29 09:30:00 | 852.00 | 2026-01-30 12:15:00 | 861.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-03 10:45:00 | 820.45 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-02-03 12:30:00 | 822.05 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-02-03 15:00:00 | 822.00 | 2026-02-10 09:15:00 | 816.75 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2026-02-04 09:15:00 | 816.10 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2026-02-09 11:30:00 | 799.80 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-02-09 13:15:00 | 799.35 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-02-09 14:30:00 | 799.55 | 2026-02-10 10:15:00 | 810.05 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-18 10:15:00 | 741.75 | 2026-02-18 13:15:00 | 762.20 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-02-24 09:15:00 | 718.05 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-24 10:45:00 | 726.40 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-25 11:00:00 | 726.20 | 2026-02-26 09:15:00 | 731.05 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-02-26 10:45:00 | 726.80 | 2026-02-27 12:15:00 | 729.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-04-13 10:15:00 | 726.50 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2026-04-13 13:30:00 | 725.95 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2026-04-15 09:15:00 | 733.50 | 2026-04-20 15:15:00 | 740.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2026-04-30 12:15:00 | 814.80 | 2026-05-08 13:15:00 | 896.28 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-05-05 11:00:00 | 813.70 | 2026-05-08 13:15:00 | 895.07 | TARGET_HIT | 1.00 | 10.00% |
