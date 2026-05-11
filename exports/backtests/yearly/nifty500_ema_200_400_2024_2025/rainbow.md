# Rainbow Childrens Medicare Ltd. (RAINBOW)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1311.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 59 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 36 |
| PARTIAL | 11 |
| TARGET_HIT | 11 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 21
- **Target hits / Stop hits / Partials:** 11 / 25 / 11
- **Avg / median % per leg:** 2.56% / 0.56%
- **Sum % (uncompounded):** 120.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 4 | 20.0% | 4 | 16 | 0 | 0.58% | 11.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 4 | 20.0% | 4 | 16 | 0 | 0.58% | 11.5% |
| SELL (all) | 27 | 22 | 81.5% | 7 | 9 | 11 | 4.04% | 109.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 22 | 81.5% | 7 | 9 | 11 | 4.04% | 109.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 26 | 55.3% | 11 | 25 | 11 | 2.56% | 120.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 14:15:00 | 1259.00 | 1298.20 | 1298.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 10:15:00 | 1254.30 | 1297.07 | 1297.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 1290.55 | 1289.35 | 1293.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 12:15:00 | 1290.55 | 1289.35 | 1293.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1290.55 | 1289.35 | 1293.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 1290.55 | 1289.35 | 1293.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 1289.85 | 1289.36 | 1293.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:30:00 | 1292.40 | 1289.36 | 1293.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1293.90 | 1289.40 | 1293.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 1295.00 | 1289.40 | 1293.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1297.00 | 1289.48 | 1293.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1322.85 | 1289.48 | 1293.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1319.00 | 1289.77 | 1293.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1294.40 | 1291.44 | 1294.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 09:45:00 | 1295.90 | 1291.44 | 1294.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 1292.00 | 1291.32 | 1294.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 13:15:00 | 1231.11 | 1282.15 | 1288.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 14:15:00 | 1229.68 | 1281.63 | 1288.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 09:15:00 | 1227.40 | 1280.54 | 1287.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-18 09:15:00 | 1164.96 | 1270.18 | 1282.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1299.65 | 1240.20 | 1240.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 1309.85 | 1240.89 | 1240.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 10:15:00 | 1365.00 | 1381.93 | 1338.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:45:00 | 1372.95 | 1381.93 | 1338.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 1340.70 | 1381.04 | 1339.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 1340.70 | 1381.04 | 1339.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 1365.85 | 1380.89 | 1339.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 1374.15 | 1380.89 | 1339.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1373.45 | 1380.82 | 1339.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 10:30:00 | 1378.80 | 1380.78 | 1339.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 11:00:00 | 1376.60 | 1380.78 | 1339.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 11:45:00 | 1380.65 | 1380.73 | 1339.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 12:30:00 | 1382.05 | 1380.71 | 1339.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-29 09:15:00 | 1516.68 | 1387.94 | 1348.40 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 09:15:00 | 1409.75 | 1518.57 | 1519.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1369.05 | 1509.95 | 1514.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1317.85 | 1312.62 | 1367.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 11:45:00 | 1316.00 | 1312.62 | 1367.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 1357.70 | 1311.79 | 1358.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 14:30:00 | 1360.00 | 1311.79 | 1358.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 1360.00 | 1312.27 | 1358.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 1337.00 | 1312.27 | 1358.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1336.40 | 1312.51 | 1358.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 1330.40 | 1315.01 | 1357.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:45:00 | 1329.85 | 1315.19 | 1357.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 15:00:00 | 1328.25 | 1315.32 | 1357.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 1326.35 | 1315.63 | 1357.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1356.60 | 1316.29 | 1357.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 1350.00 | 1316.29 | 1357.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1355.95 | 1316.68 | 1357.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:15:00 | 1353.80 | 1316.68 | 1357.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1362.50 | 1317.14 | 1357.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1362.50 | 1317.14 | 1357.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1370.10 | 1317.67 | 1357.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:45:00 | 1375.15 | 1317.67 | 1357.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1388.15 | 1319.03 | 1357.65 | SL hit (close>static) qty=1.00 sl=1387.75 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 13:15:00 | 1544.70 | 1381.80 | 1381.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 1573.00 | 1383.70 | 1382.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1398.60 | 1415.38 | 1400.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 1398.60 | 1415.38 | 1400.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1398.60 | 1415.38 | 1400.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1398.60 | 1415.38 | 1400.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1394.70 | 1415.17 | 1400.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 1395.40 | 1415.17 | 1400.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1391.40 | 1414.94 | 1400.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:15:00 | 1392.40 | 1414.94 | 1400.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:45:00 | 1397.00 | 1414.78 | 1400.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 13:45:00 | 1393.90 | 1414.51 | 1400.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 14:15:00 | 1395.30 | 1414.51 | 1400.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 1399.90 | 1414.36 | 1400.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-28 13:15:00 | 1378.30 | 1412.90 | 1400.34 | SL hit (close<static) qty=1.00 sl=1383.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1320.50 | 1390.51 | 1390.83 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 09:15:00 | 1426.60 | 1385.73 | 1385.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-12 12:15:00 | 1445.80 | 1392.16 | 1389.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 14:15:00 | 1519.40 | 1519.95 | 1484.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 1519.40 | 1519.95 | 1484.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1493.00 | 1519.34 | 1487.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1493.00 | 1519.34 | 1487.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1492.30 | 1519.07 | 1487.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1485.70 | 1519.07 | 1487.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1484.40 | 1518.73 | 1487.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1484.40 | 1518.73 | 1487.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1482.00 | 1518.36 | 1487.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 1481.10 | 1518.36 | 1487.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1479.10 | 1517.47 | 1487.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1479.10 | 1517.47 | 1487.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1480.10 | 1517.09 | 1487.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 1480.20 | 1517.09 | 1487.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1481.90 | 1515.40 | 1487.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 1478.50 | 1515.40 | 1487.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1466.70 | 1514.91 | 1487.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1466.70 | 1514.91 | 1487.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1480.00 | 1513.63 | 1487.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:45:00 | 1477.70 | 1513.63 | 1487.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1484.60 | 1513.34 | 1487.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:15:00 | 1479.90 | 1513.34 | 1487.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1469.30 | 1512.16 | 1487.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 15:15:00 | 1488.90 | 1510.17 | 1486.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:00:00 | 1481.70 | 1509.67 | 1486.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1462.20 | 1511.75 | 1503.57 | SL hit (close<static) qty=1.00 sl=1465.20 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1450.30 | 1496.51 | 1496.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1430.50 | 1494.93 | 1495.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1378.30 | 1375.49 | 1412.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:30:00 | 1385.60 | 1375.49 | 1412.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1368.00 | 1349.88 | 1369.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 1368.90 | 1349.88 | 1369.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1378.50 | 1350.17 | 1369.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 1378.50 | 1350.17 | 1369.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1377.00 | 1350.43 | 1369.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 1375.20 | 1350.43 | 1369.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1354.00 | 1352.88 | 1369.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 1363.10 | 1352.88 | 1369.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1349.90 | 1353.00 | 1369.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1342.30 | 1352.89 | 1369.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1340.00 | 1345.54 | 1362.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:00:00 | 1346.00 | 1347.20 | 1362.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:00:00 | 1346.10 | 1347.19 | 1362.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 1278.70 | 1331.69 | 1349.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 09:15:00 | 1278.79 | 1331.69 | 1349.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1275.18 | 1330.70 | 1348.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:15:00 | 1273.00 | 1330.70 | 1348.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 13:15:00 | 1208.07 | 1301.09 | 1328.37 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 1302.20 | 1213.02 | 1212.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 10:15:00 | 1331.70 | 1241.28 | 1229.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-03 09:15:00 | 1322.75 | 2024-06-03 09:15:00 | 1290.25 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-06-06 11:00:00 | 1312.50 | 2024-06-06 11:15:00 | 1296.05 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-06-07 09:15:00 | 1321.25 | 2024-06-11 14:15:00 | 1296.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1334.35 | 2024-06-11 14:15:00 | 1296.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1320.00 | 2024-06-20 13:15:00 | 1294.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-06-13 10:00:00 | 1317.30 | 2024-06-20 13:15:00 | 1294.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-06-13 12:15:00 | 1312.95 | 2024-06-20 13:15:00 | 1294.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-06-13 15:00:00 | 1314.50 | 2024-06-20 13:15:00 | 1294.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1294.40 | 2024-07-12 13:15:00 | 1231.11 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-07-05 09:45:00 | 1295.90 | 2024-07-12 14:15:00 | 1229.68 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2024-07-05 15:15:00 | 1292.00 | 2024-07-15 09:15:00 | 1227.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1294.40 | 2024-07-18 09:15:00 | 1164.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-05 09:45:00 | 1295.90 | 2024-07-18 09:15:00 | 1166.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-07-05 15:15:00 | 1292.00 | 2024-07-18 09:15:00 | 1162.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-09-06 10:30:00 | 1299.60 | 2024-09-10 10:15:00 | 1299.65 | STOP_HIT | 1.00 | -0.00% |
| BUY | retest2 | 2024-10-23 10:30:00 | 1378.80 | 2024-10-29 09:15:00 | 1516.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 11:00:00 | 1376.60 | 2024-10-29 09:15:00 | 1514.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 11:45:00 | 1380.65 | 2024-10-31 09:15:00 | 1518.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-10-23 12:30:00 | 1382.05 | 2024-10-31 09:15:00 | 1520.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-03 09:15:00 | 1567.35 | 2025-01-10 13:15:00 | 1522.30 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-01-06 09:15:00 | 1564.50 | 2025-01-10 13:15:00 | 1522.30 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-03-26 13:00:00 | 1330.40 | 2025-03-28 09:15:00 | 1388.15 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-03-26 13:45:00 | 1329.85 | 2025-03-28 09:15:00 | 1388.15 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2025-03-26 15:00:00 | 1328.25 | 2025-03-28 09:15:00 | 1388.15 | STOP_HIT | 1.00 | -4.51% |
| SELL | retest2 | 2025-03-27 09:15:00 | 1326.35 | 2025-03-28 09:15:00 | 1388.15 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2025-04-25 12:15:00 | 1392.40 | 2025-04-28 13:15:00 | 1378.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-04-25 12:45:00 | 1397.00 | 2025-04-28 13:15:00 | 1378.30 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-04-25 13:45:00 | 1393.90 | 2025-04-28 13:15:00 | 1378.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-04-25 14:15:00 | 1395.30 | 2025-04-28 13:15:00 | 1378.30 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-08-08 15:15:00 | 1488.90 | 2025-09-12 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-08-11 10:00:00 | 1481.70 | 2025-09-12 11:15:00 | 1462.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1342.30 | 2026-01-08 09:15:00 | 1278.70 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1340.00 | 2026-01-08 09:15:00 | 1278.79 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2025-12-26 10:00:00 | 1346.00 | 2026-01-08 11:15:00 | 1275.18 | PARTIAL | 0.50 | 5.26% |
| SELL | retest2 | 2025-12-26 11:00:00 | 1346.10 | 2026-01-08 11:15:00 | 1273.00 | PARTIAL | 0.50 | 5.43% |
| SELL | retest2 | 2025-12-16 11:00:00 | 1342.30 | 2026-01-20 13:15:00 | 1208.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1340.00 | 2026-01-20 13:15:00 | 1206.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-26 10:00:00 | 1346.00 | 2026-01-20 13:15:00 | 1211.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-26 11:00:00 | 1346.10 | 2026-01-20 13:15:00 | 1211.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1180.50 | 2026-03-16 09:15:00 | 1121.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1178.80 | 2026-03-16 09:15:00 | 1119.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1182.10 | 2026-03-16 09:15:00 | 1122.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-16 09:15:00 | 1117.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1180.50 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.56% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1178.80 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1182.10 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.69% |
| SELL | retest2 | 2026-03-09 15:15:00 | 1176.30 | 2026-03-25 09:15:00 | 1173.90 | STOP_HIT | 0.50 | 0.20% |
