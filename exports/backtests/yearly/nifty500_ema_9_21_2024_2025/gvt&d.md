# GE Vernova T&D India Ltd. (GVT&D)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3704 bars)
- **Last close:** 4630.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 102 |
| ALERT2 | 102 |
| ALERT2_SKIP | 60 |
| ALERT3 | 277 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 121 |
| PARTIAL | 13 |
| TARGET_HIT | 10 |
| STOP_HIT | 113 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 52 / 84
- **Target hits / Stop hits / Partials:** 10 / 113 / 13
- **Avg / median % per leg:** 0.22% / -1.05%
- **Sum % (uncompounded):** 29.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 23 | 35.9% | 8 | 55 | 1 | 0.15% | 9.3% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.68% | 11.1% |
| BUY @ 3rd Alert (retest2) | 61 | 21 | 34.4% | 7 | 54 | 0 | -0.03% | -1.7% |
| SELL (all) | 72 | 29 | 40.3% | 2 | 58 | 12 | 0.29% | 20.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 72 | 29 | 40.3% | 2 | 58 | 12 | 0.29% | 20.6% |
| retest1 (combined) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.68% | 11.1% |
| retest2 (combined) | 133 | 50 | 37.6% | 9 | 112 | 12 | 0.14% | 18.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 1043.95 | 1024.75 | 1023.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1056.05 | 1033.45 | 1027.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 12:15:00 | 1289.00 | 1305.93 | 1257.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 13:00:00 | 1289.00 | 1305.93 | 1257.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 1335.00 | 1311.74 | 1264.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 13:30:00 | 1283.95 | 1311.74 | 1264.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1369.80 | 1389.35 | 1364.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:30:00 | 1358.45 | 1389.35 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1359.60 | 1383.40 | 1363.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 15:00:00 | 1359.60 | 1383.40 | 1363.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 15:15:00 | 1358.45 | 1378.41 | 1363.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:15:00 | 1310.00 | 1378.41 | 1363.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1321.95 | 1367.12 | 1359.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1315.65 | 1367.12 | 1359.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 1395.10 | 1372.06 | 1363.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:30:00 | 1397.90 | 1372.06 | 1363.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 1385.00 | 1374.65 | 1365.38 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 1339.50 | 1361.85 | 1362.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 1325.00 | 1340.12 | 1349.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 1276.45 | 1272.76 | 1293.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 13:45:00 | 1268.35 | 1272.76 | 1293.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 1296.50 | 1277.51 | 1293.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 1296.50 | 1277.51 | 1293.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1262.55 | 1274.52 | 1291.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1312.80 | 1278.81 | 1291.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1321.00 | 1287.25 | 1294.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 1321.00 | 1287.25 | 1294.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 1331.95 | 1303.34 | 1300.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 1361.15 | 1323.17 | 1310.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1347.75 | 1388.22 | 1365.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:45:00 | 1347.75 | 1388.22 | 1365.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1347.75 | 1380.13 | 1364.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:30:00 | 1347.75 | 1380.13 | 1364.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 1280.40 | 1344.89 | 1351.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-06 12:15:00 | 1260.05 | 1291.54 | 1311.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-06 15:15:00 | 1279.00 | 1276.49 | 1298.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 09:15:00 | 1300.00 | 1281.19 | 1298.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1300.00 | 1281.19 | 1298.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:45:00 | 1306.00 | 1281.19 | 1298.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 10:15:00 | 1320.10 | 1288.97 | 1300.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:45:00 | 1320.10 | 1288.97 | 1300.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 1320.10 | 1295.20 | 1302.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:30:00 | 1320.10 | 1295.20 | 1302.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 14:15:00 | 1320.10 | 1307.35 | 1306.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 1386.10 | 1325.14 | 1314.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 15:15:00 | 1590.00 | 1590.68 | 1551.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1624.15 | 1590.68 | 1551.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 10:15:00 | 1585.00 | 1594.82 | 1560.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 11:00:00 | 1585.00 | 1594.82 | 1560.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 1565.00 | 1578.99 | 1563.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-18 15:15:00 | 1560.00 | 1575.19 | 1562.76 | SL hit (close<ema400) qty=1.00 sl=1562.76 alert=retest1 |

### Cycle 6 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1550.00 | 1557.10 | 1557.80 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 10:15:00 | 1604.00 | 1566.47 | 1561.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 12:15:00 | 1621.00 | 1584.50 | 1571.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 1587.15 | 1590.81 | 1576.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 1587.15 | 1590.81 | 1576.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1593.00 | 1591.24 | 1578.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 1596.30 | 1592.70 | 1580.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 1596.00 | 1592.68 | 1581.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:00:00 | 1597.85 | 1589.89 | 1585.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 11:15:00 | 1595.00 | 1589.93 | 1585.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 1575.00 | 1586.95 | 1584.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:30:00 | 1557.85 | 1586.95 | 1584.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1574.40 | 1584.44 | 1583.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 1582.90 | 1584.44 | 1583.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-24 13:15:00 | 1561.40 | 1579.83 | 1581.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 1561.40 | 1579.83 | 1581.58 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1597.85 | 1583.77 | 1582.73 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 1571.50 | 1580.73 | 1581.49 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 1599.00 | 1584.38 | 1583.08 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 1560.05 | 1578.30 | 1580.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 11:15:00 | 1549.95 | 1572.63 | 1577.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1555.95 | 1546.30 | 1558.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 12:00:00 | 1555.95 | 1546.30 | 1558.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 1555.70 | 1548.18 | 1557.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 14:30:00 | 1524.95 | 1539.83 | 1552.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 09:15:00 | 1501.85 | 1541.21 | 1551.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-28 15:15:00 | 1448.70 | 1503.74 | 1523.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 15:15:00 | 1519.00 | 1503.74 | 1523.47 | SL hit (close>static) qty=0.50 sl=1503.74 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1557.50 | 1535.54 | 1534.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 1613.50 | 1563.12 | 1548.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1668.00 | 1680.81 | 1651.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:00:00 | 1685.00 | 1681.65 | 1654.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 1685.60 | 1682.44 | 1657.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:00:00 | 1680.00 | 1686.14 | 1669.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 1693.40 | 1683.28 | 1682.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 1662.85 | 1679.20 | 1680.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 1662.85 | 1679.20 | 1680.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 11:15:00 | 1650.00 | 1672.88 | 1677.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1692.60 | 1666.01 | 1671.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 1692.60 | 1666.01 | 1671.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 1674.70 | 1667.74 | 1671.37 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 1698.80 | 1676.89 | 1674.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 09:15:00 | 1720.80 | 1697.25 | 1687.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 10:15:00 | 1693.00 | 1696.40 | 1688.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 11:00:00 | 1693.00 | 1696.40 | 1688.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1679.55 | 1693.03 | 1687.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:00:00 | 1679.55 | 1693.03 | 1687.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 12:15:00 | 1699.95 | 1694.42 | 1688.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 12:30:00 | 1674.35 | 1694.42 | 1688.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 13:15:00 | 1678.80 | 1691.29 | 1687.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:15:00 | 1672.00 | 1691.29 | 1687.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 1675.25 | 1688.08 | 1686.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:15:00 | 1680.00 | 1688.08 | 1686.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 1680.00 | 1686.47 | 1686.06 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1660.80 | 1681.33 | 1683.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 12:15:00 | 1640.20 | 1666.52 | 1675.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 1467.65 | 1450.60 | 1478.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:30:00 | 1448.75 | 1450.60 | 1478.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1462.70 | 1453.02 | 1477.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 14:00:00 | 1445.10 | 1456.17 | 1473.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 15:15:00 | 1445.60 | 1457.12 | 1471.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 1525.60 | 1464.45 | 1471.08 | SL hit (close>static) qty=1.00 sl=1479.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 1531.00 | 1477.76 | 1476.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1574.05 | 1526.89 | 1504.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 1572.05 | 1572.79 | 1549.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 14:00:00 | 1572.05 | 1572.79 | 1549.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1570.05 | 1572.15 | 1554.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 1572.95 | 1572.15 | 1554.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1591.75 | 1576.07 | 1558.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 1579.95 | 1576.07 | 1558.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1575.00 | 1578.07 | 1562.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:30:00 | 1570.00 | 1578.07 | 1562.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1624.00 | 1588.84 | 1570.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:45:00 | 1584.00 | 1588.84 | 1570.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 1580.00 | 1592.86 | 1575.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:15:00 | 1592.00 | 1592.86 | 1575.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 1569.65 | 1588.22 | 1574.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 1569.65 | 1588.22 | 1574.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 1570.00 | 1584.57 | 1574.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 1563.00 | 1584.57 | 1574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1575.00 | 1582.66 | 1574.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 12:30:00 | 1569.00 | 1582.66 | 1574.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1574.00 | 1580.93 | 1574.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:15:00 | 1566.95 | 1580.93 | 1574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1581.10 | 1580.96 | 1575.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:45:00 | 1567.50 | 1580.96 | 1575.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1798.00 | 1740.88 | 1692.57 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 1661.00 | 1684.47 | 1687.54 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1716.00 | 1692.37 | 1690.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 12:15:00 | 1748.95 | 1703.69 | 1695.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 1723.00 | 1729.04 | 1712.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 11:15:00 | 1767.00 | 1731.72 | 1715.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 11:30:00 | 1759.45 | 1745.24 | 1732.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 13:45:00 | 1769.00 | 1751.71 | 1737.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 1713.65 | 1733.53 | 1734.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 12:15:00 | 1713.65 | 1733.53 | 1734.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 1690.00 | 1721.86 | 1728.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1724.00 | 1718.78 | 1725.99 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 1749.90 | 1711.74 | 1709.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1826.85 | 1749.94 | 1730.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1809.00 | 1829.38 | 1800.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1809.00 | 1829.38 | 1800.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1717.50 | 1803.90 | 1793.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:30:00 | 1723.90 | 1803.90 | 1793.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 1717.50 | 1786.62 | 1786.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 1656.70 | 1723.46 | 1751.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1698.00 | 1668.96 | 1702.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 1705.05 | 1668.96 | 1702.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1705.00 | 1676.17 | 1702.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1700.00 | 1676.17 | 1702.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1717.85 | 1684.50 | 1704.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 11:30:00 | 1719.35 | 1684.50 | 1704.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 1716.00 | 1690.80 | 1705.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:30:00 | 1717.70 | 1690.80 | 1705.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 1719.30 | 1696.50 | 1706.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:45:00 | 1719.25 | 1696.50 | 1706.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1709.65 | 1699.13 | 1706.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:30:00 | 1719.35 | 1699.13 | 1706.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1719.35 | 1703.18 | 1707.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1700.00 | 1703.18 | 1707.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1725.30 | 1708.20 | 1709.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1725.30 | 1708.20 | 1709.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1699.50 | 1706.46 | 1708.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 1721.50 | 1706.46 | 1708.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1700.00 | 1705.17 | 1707.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1690.00 | 1705.17 | 1707.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 10:30:00 | 1695.00 | 1691.36 | 1698.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 1695.00 | 1689.00 | 1692.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 14:15:00 | 1699.85 | 1695.17 | 1694.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 14:15:00 | 1699.85 | 1695.17 | 1694.73 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 1676.00 | 1691.31 | 1693.05 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1732.15 | 1681.73 | 1678.30 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 1660.95 | 1684.53 | 1685.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 1644.90 | 1674.28 | 1680.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1640.30 | 1639.62 | 1656.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 1649.50 | 1639.62 | 1656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1641.00 | 1635.49 | 1645.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 1630.00 | 1635.49 | 1645.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 14:15:00 | 1548.50 | 1613.30 | 1630.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-09 14:15:00 | 1625.00 | 1604.93 | 1615.93 | SL hit (close>ema200) qty=0.50 sl=1604.93 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 11:15:00 | 1665.85 | 1628.94 | 1624.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 1675.35 | 1638.23 | 1629.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 13:15:00 | 1729.20 | 1729.69 | 1704.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 13:45:00 | 1728.25 | 1729.69 | 1704.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1700.00 | 1722.74 | 1707.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 1700.00 | 1722.74 | 1707.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1697.00 | 1717.60 | 1706.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:45:00 | 1698.05 | 1717.60 | 1706.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1690.60 | 1710.25 | 1705.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:00:00 | 1690.60 | 1710.25 | 1705.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1715.60 | 1711.32 | 1705.97 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 1690.00 | 1701.96 | 1703.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1655.20 | 1684.13 | 1692.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1696.75 | 1675.75 | 1683.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 1696.75 | 1675.75 | 1683.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1692.95 | 1679.19 | 1684.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 1670.00 | 1679.19 | 1684.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1672.15 | 1675.42 | 1681.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:30:00 | 1687.60 | 1675.42 | 1681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1671.00 | 1672.70 | 1679.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 1689.00 | 1672.70 | 1679.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1686.50 | 1671.83 | 1677.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 1686.50 | 1671.83 | 1677.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1700.00 | 1677.46 | 1679.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1606.85 | 1677.46 | 1679.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-20 09:15:00 | 1526.51 | 1620.31 | 1641.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 1553.55 | 1551.04 | 1564.86 | SL hit (close>ema200) qty=0.50 sl=1551.04 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 12:15:00 | 1600.05 | 1574.96 | 1573.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 13:15:00 | 1620.70 | 1584.11 | 1577.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1589.00 | 1598.36 | 1587.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 1578.55 | 1598.36 | 1587.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1600.50 | 1598.79 | 1588.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:00:00 | 1615.35 | 1602.10 | 1590.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 13:30:00 | 1613.95 | 1606.76 | 1594.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:15:00 | 1626.00 | 1611.75 | 1600.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 1629.95 | 1651.67 | 1652.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 1629.95 | 1651.67 | 1652.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 1677.00 | 1654.65 | 1653.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 10:15:00 | 1689.00 | 1661.52 | 1656.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1655.00 | 1661.57 | 1657.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 13:00:00 | 1655.00 | 1661.57 | 1657.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1660.85 | 1661.43 | 1658.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 1660.85 | 1661.43 | 1658.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1656.40 | 1660.42 | 1657.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:30:00 | 1660.00 | 1660.42 | 1657.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1660.00 | 1660.34 | 1658.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1686.00 | 1660.34 | 1658.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1667.45 | 1661.76 | 1658.96 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1596.50 | 1648.71 | 1653.28 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1685.70 | 1644.12 | 1642.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 1711.55 | 1673.70 | 1657.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 15:15:00 | 1756.50 | 1763.23 | 1732.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:15:00 | 1786.55 | 1763.23 | 1732.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1779.00 | 1766.39 | 1736.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 1794.80 | 1766.39 | 1736.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1767.75 | 1771.80 | 1753.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 1812.00 | 1771.80 | 1753.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 1780.00 | 1775.17 | 1758.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 14:15:00 | 1875.88 | 1823.08 | 1798.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-10-17 09:15:00 | 1965.21 | 1911.17 | 1871.28 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 34 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 1813.40 | 1864.38 | 1868.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 1790.00 | 1830.23 | 1848.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1764.85 | 1746.90 | 1771.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1764.85 | 1746.90 | 1771.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1794.95 | 1756.51 | 1773.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:00:00 | 1794.95 | 1756.51 | 1773.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1794.95 | 1764.20 | 1775.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 1798.00 | 1764.20 | 1775.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1768.60 | 1767.61 | 1775.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:30:00 | 1790.00 | 1767.61 | 1775.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1758.90 | 1760.69 | 1769.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:00:00 | 1758.90 | 1760.69 | 1769.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 1750.00 | 1758.55 | 1767.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 11:45:00 | 1771.20 | 1758.55 | 1767.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 14:15:00 | 1676.25 | 1655.87 | 1684.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 15:00:00 | 1676.25 | 1655.87 | 1684.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 15:15:00 | 1659.00 | 1656.50 | 1681.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:15:00 | 1635.85 | 1656.50 | 1681.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:00:00 | 1627.00 | 1639.84 | 1664.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 1689.85 | 1651.47 | 1665.93 | SL hit (close>static) qty=1.00 sl=1683.95 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1706.15 | 1675.17 | 1674.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 12:15:00 | 1720.15 | 1690.35 | 1681.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1743.00 | 1772.68 | 1746.09 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 10:15:00 | 1707.80 | 1733.60 | 1735.46 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 1741.10 | 1737.14 | 1736.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 10:15:00 | 1777.00 | 1749.54 | 1742.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 1735.25 | 1757.05 | 1751.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 11:30:00 | 1750.00 | 1757.05 | 1751.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1690.00 | 1743.64 | 1746.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 1675.00 | 1715.82 | 1731.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1777.15 | 1728.09 | 1735.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:30:00 | 1777.15 | 1728.09 | 1735.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 1777.15 | 1745.75 | 1742.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 09:15:00 | 1866.00 | 1784.63 | 1763.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1813.05 | 1838.35 | 1808.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:30:00 | 1816.00 | 1838.35 | 1808.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 1819.95 | 1834.67 | 1809.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:30:00 | 1819.95 | 1834.67 | 1809.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 1799.95 | 1827.72 | 1808.93 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 09:15:00 | 1779.95 | 1797.20 | 1799.41 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 14:15:00 | 1854.85 | 1785.55 | 1785.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1933.00 | 1825.35 | 1804.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 1859.80 | 1886.44 | 1870.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:45:00 | 1866.95 | 1886.44 | 1870.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1900.00 | 1889.15 | 1872.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 11:30:00 | 1905.50 | 1890.13 | 1874.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 09:15:00 | 1918.00 | 1885.79 | 1877.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 11:45:00 | 1909.10 | 1901.88 | 1887.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 1823.50 | 1890.16 | 1888.54 | SL hit (close<static) qty=1.00 sl=1859.80 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 10:15:00 | 1823.50 | 1876.83 | 1882.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 1738.00 | 1820.38 | 1849.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 1799.75 | 1797.98 | 1824.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:00:00 | 1799.75 | 1797.98 | 1824.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1765.65 | 1752.10 | 1761.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1765.65 | 1752.10 | 1761.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1774.00 | 1756.48 | 1762.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:45:00 | 1787.95 | 1763.16 | 1765.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1800.00 | 1770.53 | 1768.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1810.90 | 1778.60 | 1772.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1859.95 | 1861.72 | 1835.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 1854.50 | 1861.72 | 1835.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1904.00 | 1890.30 | 1865.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1918.90 | 1897.86 | 1883.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 11:00:00 | 1924.00 | 1930.06 | 1921.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 1909.25 | 1926.05 | 1928.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1909.25 | 1926.05 | 1928.01 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 1935.00 | 1929.37 | 1929.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 1980.00 | 1939.50 | 1933.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 09:15:00 | 2034.00 | 2056.49 | 2024.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 10:00:00 | 2034.00 | 2056.49 | 2024.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2049.95 | 2055.18 | 2026.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:15:00 | 2048.80 | 2055.18 | 2026.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 2050.00 | 2054.15 | 2029.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 2062.65 | 2052.16 | 2037.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:45:00 | 2061.95 | 2053.82 | 2040.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:45:00 | 2061.50 | 2055.34 | 2042.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 2060.15 | 2056.09 | 2043.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 2134.95 | 2071.87 | 2052.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 2058.00 | 2071.87 | 2052.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 2100.00 | 2128.63 | 2102.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 2111.00 | 2128.63 | 2102.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2109.95 | 2124.90 | 2103.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:30:00 | 2100.50 | 2124.90 | 2103.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 2099.65 | 2119.85 | 2103.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 2093.90 | 2119.85 | 2103.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 2113.20 | 2118.52 | 2103.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 13:30:00 | 2127.30 | 2121.01 | 2106.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 2094.00 | 2111.22 | 2105.15 | SL hit (close<static) qty=1.00 sl=2099.65 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 12:15:00 | 2080.05 | 2097.59 | 2099.71 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 2088.40 | 2063.79 | 2062.01 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 2018.00 | 2060.75 | 2063.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 2000.60 | 2032.63 | 2045.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 2030.60 | 2027.54 | 2040.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 2030.60 | 2027.54 | 2040.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 2039.00 | 2025.62 | 2037.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 2043.50 | 2025.62 | 2037.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2039.05 | 2028.30 | 2037.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:30:00 | 2049.45 | 2028.30 | 2037.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2039.50 | 2030.54 | 2037.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2039.95 | 2030.54 | 2037.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2087.50 | 2041.93 | 2042.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 2087.50 | 2041.93 | 2042.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 10:15:00 | 2065.00 | 2046.55 | 2044.38 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2006.90 | 2042.95 | 2043.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 1990.05 | 2016.40 | 2028.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2016.00 | 2010.70 | 2020.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1967.95 | 1994.66 | 2001.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 1970.80 | 1972.79 | 1986.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 15:15:00 | 1951.00 | 1971.37 | 1983.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1869.55 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1872.26 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-14 09:15:00 | 1853.45 | 1907.32 | 1935.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 1886.85 | 1882.50 | 1910.89 | SL hit (close>ema200) qty=0.50 sl=1882.50 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1925.40 | 1910.84 | 1909.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 1980.00 | 1926.22 | 1917.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 1946.50 | 1951.73 | 1935.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-17 12:00:00 | 1946.50 | 1951.73 | 1935.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1930.10 | 1945.85 | 1935.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:00:00 | 1930.10 | 1945.85 | 1935.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1945.90 | 1945.86 | 1936.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:30:00 | 1931.35 | 1945.86 | 1936.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1932.10 | 1943.11 | 1936.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 1900.05 | 1943.11 | 1936.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1919.90 | 1938.46 | 1934.71 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 10:15:00 | 1900.00 | 1930.77 | 1931.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 11:15:00 | 1891.05 | 1922.83 | 1927.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1787.00 | 1779.04 | 1819.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1787.00 | 1779.04 | 1819.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1834.45 | 1790.12 | 1820.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 1834.45 | 1790.12 | 1820.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1833.80 | 1798.86 | 1822.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:30:00 | 1800.65 | 1799.09 | 1820.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:30:00 | 1799.10 | 1798.08 | 1813.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1710.62 | 1757.01 | 1783.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1709.14 | 1757.01 | 1783.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 09:15:00 | 1620.59 | 1675.68 | 1723.36 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1709.90 | 1663.77 | 1659.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1760.30 | 1706.37 | 1683.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1724.90 | 1763.68 | 1738.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 1724.25 | 1763.68 | 1738.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1698.15 | 1750.57 | 1735.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 14:00:00 | 1698.15 | 1750.57 | 1735.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1656.15 | 1719.84 | 1723.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 11:15:00 | 1640.30 | 1692.81 | 1710.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 1646.00 | 1632.60 | 1659.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 1646.00 | 1632.60 | 1659.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1697.00 | 1647.94 | 1661.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1705.00 | 1647.94 | 1661.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1729.25 | 1664.20 | 1667.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 1729.25 | 1664.20 | 1667.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 1729.25 | 1677.21 | 1673.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 1736.30 | 1703.83 | 1690.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 15:15:00 | 1705.00 | 1716.24 | 1703.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:15:00 | 1697.75 | 1716.24 | 1703.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1702.00 | 1713.39 | 1703.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:30:00 | 1682.80 | 1713.39 | 1703.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1704.60 | 1711.64 | 1703.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 1702.20 | 1711.64 | 1703.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1695.00 | 1708.31 | 1702.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:00:00 | 1695.00 | 1708.31 | 1702.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1695.00 | 1705.65 | 1701.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 1695.35 | 1705.65 | 1701.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1695.10 | 1701.02 | 1700.45 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1655.10 | 1691.83 | 1696.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1613.25 | 1650.22 | 1666.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1563.75 | 1559.09 | 1593.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1563.75 | 1559.09 | 1593.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1475.00 | 1541.46 | 1569.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 1414.95 | 1492.10 | 1528.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 09:15:00 | 1344.20 | 1378.85 | 1403.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 1387.85 | 1380.65 | 1402.29 | SL hit (close>ema200) qty=0.50 sl=1380.65 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 15:15:00 | 1426.65 | 1411.91 | 1411.20 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1382.65 | 1406.06 | 1408.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 1380.75 | 1401.00 | 1406.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 13:15:00 | 1399.95 | 1397.05 | 1402.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 14:15:00 | 1394.50 | 1397.05 | 1402.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1438.40 | 1405.32 | 1405.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 15:00:00 | 1438.40 | 1405.32 | 1405.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1399.70 | 1404.20 | 1405.30 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 1417.95 | 1406.95 | 1406.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-24 10:15:00 | 1468.55 | 1419.27 | 1412.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 15:15:00 | 1436.00 | 1438.80 | 1426.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 09:15:00 | 1429.15 | 1438.80 | 1426.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1444.60 | 1439.96 | 1428.29 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1399.15 | 1422.70 | 1425.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 11:15:00 | 1390.30 | 1416.22 | 1422.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1314.45 | 1313.64 | 1341.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 14:45:00 | 1316.55 | 1313.64 | 1341.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1348.95 | 1318.52 | 1338.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1348.95 | 1318.52 | 1338.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1362.00 | 1327.21 | 1341.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 1377.40 | 1327.21 | 1341.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 14:15:00 | 1362.55 | 1348.14 | 1347.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 1401.00 | 1360.29 | 1353.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 1371.30 | 1380.55 | 1371.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 1371.30 | 1380.55 | 1371.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 1376.15 | 1379.67 | 1372.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:45:00 | 1391.40 | 1381.73 | 1374.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 10:15:00 | 1394.95 | 1383.58 | 1376.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 1427.40 | 1432.51 | 1433.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 10:15:00 | 1427.40 | 1432.51 | 1433.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 12:15:00 | 1424.80 | 1430.39 | 1432.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 1428.00 | 1427.42 | 1430.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 1433.85 | 1427.42 | 1430.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1455.65 | 1433.06 | 1432.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 10:15:00 | 1471.15 | 1440.68 | 1435.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 09:15:00 | 1505.85 | 1506.70 | 1485.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 13:15:00 | 1485.50 | 1503.93 | 1491.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 1485.50 | 1503.93 | 1491.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 14:00:00 | 1485.50 | 1503.93 | 1491.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 1505.15 | 1504.18 | 1492.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 1516.35 | 1505.10 | 1494.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 1481.40 | 1500.36 | 1492.94 | SL hit (close<static) qty=1.00 sl=1485.60 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 11:15:00 | 1547.00 | 1579.12 | 1579.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 13:15:00 | 1536.50 | 1565.95 | 1573.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1583.35 | 1551.98 | 1560.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:45:00 | 1584.00 | 1551.98 | 1560.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1538.25 | 1549.23 | 1558.55 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 1605.00 | 1571.10 | 1567.54 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1519.25 | 1559.89 | 1564.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1509.55 | 1549.82 | 1559.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1503.10 | 1500.97 | 1524.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 1503.10 | 1500.97 | 1524.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1411.35 | 1449.17 | 1474.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 13:30:00 | 1409.25 | 1429.91 | 1456.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1338.79 | 1399.35 | 1435.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 15:15:00 | 1311.00 | 1309.33 | 1341.19 | SL hit (close>ema200) qty=0.50 sl=1309.33 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 15:15:00 | 1348.00 | 1325.20 | 1323.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1378.40 | 1335.84 | 1328.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1407.80 | 1411.06 | 1384.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 1410.80 | 1411.06 | 1384.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 1520.00 | 1514.05 | 1499.64 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 1488.50 | 1497.68 | 1498.78 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 1521.50 | 1501.06 | 1499.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 1535.50 | 1507.94 | 1502.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 1570.20 | 1573.84 | 1554.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 14:30:00 | 1567.90 | 1573.84 | 1554.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1548.50 | 1568.77 | 1554.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 1571.50 | 1568.77 | 1554.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:30:00 | 1572.00 | 1566.11 | 1555.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 1575.70 | 1559.43 | 1555.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 1586.00 | 1562.83 | 1558.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1604.40 | 1574.34 | 1564.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 1576.60 | 1574.34 | 1564.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 1576.90 | 1578.46 | 1570.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:30:00 | 1563.50 | 1578.46 | 1570.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1561.40 | 1575.05 | 1569.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1561.40 | 1575.05 | 1569.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1538.50 | 1567.74 | 1566.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 1538.50 | 1567.74 | 1566.61 | SL hit (close<static) qty=1.00 sl=1540.20 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 1539.30 | 1562.05 | 1564.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 1526.00 | 1554.84 | 1560.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1560.40 | 1546.53 | 1554.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:45:00 | 1562.20 | 1546.53 | 1554.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1564.20 | 1550.07 | 1555.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 1564.20 | 1550.07 | 1555.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 1592.00 | 1558.45 | 1558.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 1592.00 | 1558.45 | 1558.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1630.90 | 1572.94 | 1565.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 14:15:00 | 1656.30 | 1629.87 | 1613.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 1849.10 | 1852.09 | 1817.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 10:00:00 | 1849.10 | 1852.09 | 1817.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1845.00 | 1845.35 | 1830.63 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 1794.00 | 1822.77 | 1824.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 15:15:00 | 1785.00 | 1815.22 | 1820.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1845.00 | 1820.98 | 1822.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1845.00 | 1820.98 | 1822.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1826.40 | 1822.06 | 1822.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:15:00 | 1823.90 | 1822.71 | 1823.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 1849.90 | 1828.45 | 1825.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1870.00 | 1843.37 | 1834.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2260.30 | 2274.98 | 2226.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2246.00 | 2271.15 | 2242.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 2246.00 | 2271.15 | 2242.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2234.80 | 2263.88 | 2242.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 2309.20 | 2263.88 | 2242.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 13:15:00 | 2332.40 | 2364.69 | 2367.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 2314.90 | 2354.74 | 2362.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 15:15:00 | 2313.60 | 2311.22 | 2330.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 2320.40 | 2311.22 | 2330.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2319.80 | 2312.93 | 2329.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 2319.80 | 2312.93 | 2329.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 2397.20 | 2323.23 | 2325.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 2370.70 | 2323.23 | 2325.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 2403.00 | 2339.18 | 2332.18 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 2290.50 | 2336.67 | 2341.04 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 2366.30 | 2319.90 | 2316.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 2387.00 | 2353.32 | 2336.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 2365.40 | 2367.75 | 2349.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 2350.00 | 2367.75 | 2349.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2370.00 | 2368.20 | 2351.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:30:00 | 2392.40 | 2365.06 | 2354.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 2345.00 | 2354.60 | 2353.00 | SL hit (close<static) qty=1.00 sl=2347.60 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 09:15:00 | 2326.40 | 2348.96 | 2350.58 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 2408.80 | 2350.72 | 2346.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 13:15:00 | 2434.00 | 2382.42 | 2363.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 10:15:00 | 2398.30 | 2406.20 | 2383.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 11:00:00 | 2398.30 | 2406.20 | 2383.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 2395.10 | 2403.98 | 2384.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:30:00 | 2413.90 | 2403.98 | 2384.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 2382.60 | 2399.70 | 2383.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:45:00 | 2378.00 | 2399.70 | 2383.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 2387.00 | 2397.16 | 2384.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 2371.00 | 2397.16 | 2384.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2394.90 | 2396.71 | 2385.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 15:15:00 | 2390.00 | 2396.71 | 2385.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 2390.00 | 2395.37 | 2385.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 2415.60 | 2395.37 | 2385.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:45:00 | 2412.50 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 2400.00 | 2395.96 | 2387.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 2375.50 | 2391.87 | 2386.50 | SL hit (close<static) qty=1.00 sl=2381.20 alert=retest2 |

### Cycle 80 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 2354.60 | 2381.65 | 2382.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 14:15:00 | 2347.60 | 2374.84 | 2379.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2370.10 | 2368.87 | 2375.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 2362.70 | 2368.87 | 2375.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 2362.70 | 2367.64 | 2373.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:45:00 | 2370.20 | 2367.64 | 2373.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 2362.70 | 2354.14 | 2363.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:00:00 | 2362.70 | 2354.14 | 2363.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 2353.00 | 2353.91 | 2362.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2349.50 | 2357.26 | 2361.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 12:45:00 | 2348.90 | 2358.84 | 2361.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 14:15:00 | 2348.70 | 2357.07 | 2360.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 2345.60 | 2354.21 | 2358.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2355.00 | 2354.37 | 2357.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 11:15:00 | 2375.10 | 2361.35 | 2360.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 14:15:00 | 2387.20 | 2370.18 | 2365.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2353.40 | 2368.87 | 2365.53 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 2344.30 | 2360.53 | 2362.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 10:15:00 | 2332.60 | 2349.77 | 2355.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 2363.10 | 2347.10 | 2353.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 2357.60 | 2347.10 | 2353.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 2358.70 | 2349.42 | 2353.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 2370.00 | 2349.42 | 2353.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2359.10 | 2351.35 | 2354.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 15:15:00 | 2345.20 | 2351.35 | 2354.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 2350.00 | 2351.35 | 2353.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 2349.40 | 2350.60 | 2353.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2372.20 | 2350.31 | 2350.72 | SL hit (close>static) qty=1.00 sl=2364.60 alert=retest2 |

### Cycle 83 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 2383.90 | 2356.95 | 2353.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 2424.10 | 2381.47 | 2367.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 13:15:00 | 2387.30 | 2388.40 | 2375.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 13:45:00 | 2394.00 | 2388.40 | 2375.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2372.00 | 2384.73 | 2377.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 2372.00 | 2384.73 | 2377.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2400.00 | 2387.79 | 2379.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 2407.20 | 2390.93 | 2381.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 2367.00 | 2383.56 | 2379.63 | SL hit (close<static) qty=1.00 sl=2370.90 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 09:15:00 | 2332.50 | 2373.31 | 2375.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 2320.30 | 2356.51 | 2367.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2328.40 | 2299.84 | 2319.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 2322.80 | 2299.84 | 2319.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 2340.40 | 2307.95 | 2321.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 2340.40 | 2307.95 | 2321.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 2322.10 | 2310.78 | 2321.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:45:00 | 2332.90 | 2310.78 | 2321.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 2322.00 | 2313.02 | 2321.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 2325.40 | 2313.02 | 2321.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 2305.60 | 2311.54 | 2320.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:30:00 | 2319.40 | 2311.54 | 2320.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 2329.00 | 2311.39 | 2317.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:30:00 | 2319.20 | 2311.39 | 2317.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2323.50 | 2313.81 | 2318.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 2325.50 | 2313.81 | 2318.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 2333.10 | 2313.36 | 2316.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:00:00 | 2333.10 | 2313.36 | 2316.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 2340.30 | 2318.75 | 2318.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 2360.60 | 2328.82 | 2323.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 2375.00 | 2388.87 | 2363.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:45:00 | 2392.20 | 2388.87 | 2363.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2393.50 | 2389.79 | 2366.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:00:00 | 2407.90 | 2377.77 | 2368.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 12:15:00 | 2440.00 | 2474.00 | 2475.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 2434.50 | 2460.66 | 2468.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2470.00 | 2418.81 | 2430.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:30:00 | 2456.40 | 2418.81 | 2430.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2476.80 | 2430.41 | 2434.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 2485.40 | 2430.41 | 2434.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 2475.20 | 2439.37 | 2437.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 2597.00 | 2476.76 | 2455.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 09:15:00 | 2795.50 | 2820.49 | 2742.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:00:00 | 2795.50 | 2820.49 | 2742.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2759.80 | 2791.82 | 2747.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 15:15:00 | 2810.00 | 2782.87 | 2750.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 2787.90 | 2797.37 | 2798.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 2751.90 | 2788.28 | 2794.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2789.20 | 2786.85 | 2792.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 2789.20 | 2786.85 | 2792.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 2799.90 | 2789.46 | 2793.08 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 13:15:00 | 2822.10 | 2795.99 | 2795.72 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 2790.00 | 2795.43 | 2795.55 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 09:15:00 | 2815.40 | 2799.43 | 2797.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 2914.60 | 2821.96 | 2809.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 2848.90 | 2885.70 | 2857.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 2848.90 | 2885.70 | 2857.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 2890.00 | 2886.56 | 2860.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 2911.90 | 2856.92 | 2853.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 2839.70 | 2852.35 | 2851.55 | SL hit (close<static) qty=1.00 sl=2848.90 alert=retest2 |

### Cycle 92 — SELL (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 12:15:00 | 2840.00 | 2849.88 | 2850.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 2829.90 | 2844.29 | 2847.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 2821.30 | 2817.73 | 2828.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:15:00 | 2827.10 | 2817.73 | 2828.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 2792.00 | 2812.58 | 2825.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:30:00 | 2788.60 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 11:00:00 | 2785.00 | 2807.07 | 2821.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 2787.10 | 2801.61 | 2814.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:45:00 | 2789.80 | 2793.57 | 2807.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 2791.80 | 2793.21 | 2805.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 2812.50 | 2793.21 | 2805.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2787.20 | 2775.48 | 2790.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 2718.60 | 2766.63 | 2776.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 11:00:00 | 2709.60 | 2745.17 | 2764.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 2723.70 | 2698.55 | 2712.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 2781.50 | 2730.75 | 2725.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 2819.90 | 2778.26 | 2757.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 2763.60 | 2775.32 | 2757.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 10:00:00 | 2763.60 | 2775.32 | 2757.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 2783.20 | 2776.90 | 2760.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 2809.20 | 2773.48 | 2764.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 2756.10 | 2791.28 | 2783.49 | SL hit (close<static) qty=1.00 sl=2760.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 2729.10 | 2772.93 | 2776.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 2706.80 | 2752.35 | 2765.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 2756.20 | 2748.03 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 2756.20 | 2748.03 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2761.20 | 2750.66 | 2761.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:00:00 | 2761.20 | 2750.66 | 2761.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 2777.40 | 2756.01 | 2762.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:30:00 | 2776.70 | 2756.01 | 2762.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 2760.00 | 2756.81 | 2762.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 15:00:00 | 2749.20 | 2755.30 | 2760.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:45:00 | 2752.10 | 2751.98 | 2758.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 2745.00 | 2749.90 | 2756.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:15:00 | 2751.90 | 2751.01 | 2755.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 2752.00 | 2751.21 | 2755.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 2737.10 | 2748.89 | 2754.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:15:00 | 2730.70 | 2749.24 | 2753.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 2773.90 | 2747.85 | 2745.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 09:15:00 | 2828.00 | 2771.22 | 2757.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 2760.20 | 2786.45 | 2771.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 2760.20 | 2786.45 | 2771.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 2743.30 | 2777.82 | 2768.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 2743.30 | 2777.82 | 2768.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 2760.50 | 2767.14 | 2765.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:30:00 | 2752.10 | 2767.14 | 2765.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 2781.70 | 2770.05 | 2766.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 2789.40 | 2770.94 | 2767.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:45:00 | 2794.00 | 2773.87 | 2769.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 2786.00 | 2778.24 | 2772.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-19 10:15:00 | 3068.34 | 3001.25 | 2959.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 13:15:00 | 3006.30 | 3011.44 | 3011.94 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 3030.00 | 3010.39 | 3009.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 3030.30 | 3016.74 | 3012.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2998.20 | 3013.03 | 3011.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 2998.20 | 3013.03 | 3011.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 2978.10 | 3006.05 | 3008.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 13:15:00 | 2963.00 | 2997.44 | 3004.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 15:15:00 | 2946.60 | 2944.84 | 2966.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:15:00 | 2989.70 | 2944.84 | 2966.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 2959.60 | 2947.79 | 2965.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 2997.40 | 2947.79 | 2965.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2966.30 | 2951.50 | 2965.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 2953.70 | 2951.50 | 2965.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 2986.00 | 2958.40 | 2967.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:45:00 | 2988.80 | 2958.40 | 2967.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3000.00 | 2966.72 | 2970.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 3000.00 | 2966.72 | 2970.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 2988.70 | 2974.84 | 2973.63 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 10:15:00 | 2953.40 | 2972.30 | 2973.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 11:15:00 | 2945.40 | 2966.92 | 2970.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 12:15:00 | 2967.80 | 2967.09 | 2970.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 13:00:00 | 2967.80 | 2967.09 | 2970.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 2961.10 | 2965.90 | 2969.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 15:15:00 | 2944.60 | 2964.84 | 2968.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3034.40 | 2975.51 | 2972.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 3070.00 | 3015.42 | 2995.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 13:15:00 | 3133.30 | 3146.13 | 3104.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:30:00 | 3129.10 | 3146.13 | 3104.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3135.30 | 3142.89 | 3113.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 3137.70 | 3142.89 | 3113.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 3135.00 | 3139.85 | 3119.45 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 3065.10 | 3109.00 | 3110.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 13:15:00 | 3030.50 | 3082.73 | 3097.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 3049.50 | 3037.88 | 3061.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 3049.50 | 3037.88 | 3061.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3042.90 | 3039.29 | 3058.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 3049.00 | 3039.29 | 3058.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3044.50 | 3025.59 | 3039.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:30:00 | 3033.60 | 3025.59 | 3039.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 3040.00 | 3028.47 | 3039.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:00:00 | 3022.50 | 3034.38 | 3040.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:15:00 | 2968.20 | 3031.79 | 3038.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3004.10 | 2965.79 | 2963.94 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 13:15:00 | 2962.30 | 2967.16 | 2967.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 14:15:00 | 2956.00 | 2964.93 | 2966.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 2921.80 | 2897.88 | 2927.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 2910.00 | 2897.88 | 2927.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 2910.10 | 2900.32 | 2925.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 2893.50 | 2900.04 | 2923.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 2900.50 | 2903.98 | 2921.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 14:45:00 | 2902.00 | 2903.52 | 2919.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 2873.20 | 2897.12 | 2913.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2927.70 | 2871.15 | 2886.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 2932.70 | 2871.15 | 2886.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 2958.60 | 2888.64 | 2893.41 | SL hit (close>static) qty=1.00 sl=2929.90 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 11:15:00 | 2950.00 | 2900.91 | 2898.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 13:15:00 | 2968.60 | 2922.27 | 2909.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 10:15:00 | 2933.50 | 2943.78 | 2925.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:00:00 | 2933.50 | 2943.78 | 2925.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 3034.30 | 3051.27 | 3025.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 3112.80 | 3044.26 | 3028.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 3073.60 | 3117.32 | 3106.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3059.80 | 3097.86 | 3098.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3039.30 | 3086.15 | 3093.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3096.30 | 3058.60 | 3075.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3093.90 | 3058.60 | 3075.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3071.70 | 3061.22 | 3074.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:30:00 | 3087.20 | 3061.22 | 3074.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 3098.60 | 3068.70 | 3077.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 3098.60 | 3068.70 | 3077.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3104.10 | 3075.78 | 3079.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 3090.50 | 3075.42 | 3079.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3117.50 | 3084.07 | 3082.14 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 3080.40 | 3114.32 | 3116.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 3051.60 | 3092.90 | 3105.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3088.90 | 3069.70 | 3088.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:30:00 | 3110.00 | 3069.70 | 3088.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3065.60 | 3068.88 | 3086.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:15:00 | 3108.90 | 3068.88 | 3086.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3069.30 | 3068.96 | 3084.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 3074.90 | 3068.96 | 3084.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2997.80 | 3008.57 | 3034.31 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 3068.30 | 3043.25 | 3040.79 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 2976.70 | 3044.39 | 3045.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 10:15:00 | 2943.00 | 3024.11 | 3036.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 3002.30 | 2956.27 | 2988.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 3002.30 | 2956.27 | 2988.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 2974.80 | 2959.98 | 2986.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:30:00 | 3003.00 | 2959.98 | 2986.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 2972.00 | 2962.38 | 2985.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:45:00 | 2987.40 | 2962.38 | 2985.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 2989.30 | 2969.34 | 2984.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 2988.80 | 2969.34 | 2984.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2867.60 | 2948.99 | 2974.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 2843.00 | 2948.99 | 2974.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 2998.30 | 2942.99 | 2937.22 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 2884.50 | 2928.96 | 2932.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2880.00 | 2901.42 | 2915.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 10:15:00 | 2902.00 | 2897.96 | 2911.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:30:00 | 2907.40 | 2897.96 | 2911.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 2875.30 | 2890.96 | 2903.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 2892.20 | 2890.96 | 2903.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2873.80 | 2886.59 | 2899.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 2832.00 | 2874.82 | 2891.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 09:30:00 | 2841.80 | 2834.40 | 2863.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 10:15:00 | 2900.00 | 2869.12 | 2867.43 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 11:15:00 | 2827.90 | 2868.28 | 2870.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 2819.50 | 2853.25 | 2863.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 2760.00 | 2744.59 | 2774.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 2760.00 | 2744.59 | 2774.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 2836.30 | 2764.67 | 2776.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 2836.30 | 2764.67 | 2776.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2865.10 | 2784.75 | 2784.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 12:15:00 | 2895.50 | 2806.90 | 2794.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 13:15:00 | 2899.10 | 2903.68 | 2863.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 14:00:00 | 2899.10 | 2903.68 | 2863.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 2914.20 | 2905.78 | 2867.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 2935.80 | 2913.51 | 2886.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 2959.60 | 2988.00 | 2989.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 2846.40 | 2950.60 | 2970.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2931.30 | 2894.08 | 2920.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 2931.30 | 2894.08 | 2920.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 2897.50 | 2894.76 | 2918.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 2880.60 | 2894.76 | 2918.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3085.30 | 2939.40 | 2930.77 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3065.00 | 3096.24 | 3097.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 3035.40 | 3070.09 | 3081.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 3085.00 | 3073.08 | 3081.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:30:00 | 3074.60 | 3073.08 | 3081.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 3071.10 | 3072.68 | 3080.85 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 3135.60 | 3089.60 | 3086.84 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 3081.40 | 3124.30 | 3125.53 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3171.20 | 3106.31 | 3102.35 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 3087.10 | 3118.26 | 3118.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 3009.10 | 3096.43 | 3108.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2812.30 | 2755.09 | 2803.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 2813.80 | 2755.09 | 2803.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2802.70 | 2764.61 | 2803.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2787.30 | 2771.31 | 2803.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 2747.90 | 2778.83 | 2796.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 14:15:00 | 2647.93 | 2693.49 | 2742.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 2610.51 | 2675.01 | 2724.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 2575.10 | 2571.98 | 2607.73 | SL hit (close>ema200) qty=0.50 sl=2571.98 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 2646.70 | 2622.79 | 2619.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 2746.40 | 2652.67 | 2634.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 15:15:00 | 2700.00 | 2704.70 | 2674.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-27 09:15:00 | 2704.90 | 2704.70 | 2674.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2720.00 | 2707.76 | 2678.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 15:00:00 | 2733.40 | 2708.66 | 2689.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-29 09:15:00 | 3006.74 | 2880.93 | 2804.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3564.20 | 3627.06 | 3632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 3552.30 | 3595.39 | 3613.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3694.60 | 3610.48 | 3616.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 3694.60 | 3610.48 | 3616.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 3687.10 | 3625.80 | 3623.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 3702.60 | 3660.83 | 3643.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 3643.20 | 3657.31 | 3643.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:00:00 | 3643.20 | 3657.31 | 3643.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 3641.00 | 3654.05 | 3643.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:15:00 | 3626.60 | 3654.05 | 3643.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 3636.00 | 3650.44 | 3642.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:45:00 | 3621.70 | 3650.44 | 3642.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 3628.80 | 3646.11 | 3641.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:45:00 | 3626.90 | 3646.11 | 3641.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 3647.80 | 3646.45 | 3642.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 3655.00 | 3646.45 | 3642.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:45:00 | 3665.50 | 3650.96 | 3645.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 3566.50 | 3674.29 | 3665.72 | SL hit (close<static) qty=1.00 sl=3625.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 3563.00 | 3652.04 | 3656.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 3542.40 | 3630.11 | 3646.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 3579.60 | 3569.33 | 3601.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 3579.60 | 3569.33 | 3601.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 3622.40 | 3579.94 | 3603.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:45:00 | 3642.90 | 3579.94 | 3603.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3661.10 | 3596.17 | 3608.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 3662.00 | 3596.17 | 3608.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 3729.40 | 3637.54 | 3625.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 3824.10 | 3754.01 | 3715.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 3842.30 | 3849.22 | 3808.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 3865.10 | 3852.48 | 3829.81 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 3775.00 | 3820.33 | 3821.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3742.10 | 3791.38 | 3806.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3799.10 | 3721.24 | 3751.14 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 3818.20 | 3775.32 | 3770.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 3838.00 | 3796.43 | 3781.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3745.60 | 3845.48 | 3828.37 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 3697.00 | 3815.78 | 3816.43 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 3878.60 | 3809.72 | 3807.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 3885.60 | 3824.89 | 3814.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 3794.80 | 3835.93 | 3827.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 3794.80 | 3835.93 | 3827.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 3779.90 | 3824.73 | 3823.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 3779.90 | 3824.73 | 3823.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 3756.90 | 3811.16 | 3817.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 3749.90 | 3782.49 | 3800.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3619.00 | 3543.16 | 3597.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 3619.00 | 3543.16 | 3597.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3668.30 | 3568.18 | 3604.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 3668.30 | 3568.18 | 3604.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 3663.20 | 3625.82 | 3625.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 3769.70 | 3668.41 | 3646.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 13:15:00 | 3766.80 | 3779.22 | 3739.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:00:00 | 3766.80 | 3779.22 | 3739.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 3768.40 | 3777.05 | 3742.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:45:00 | 3734.40 | 3777.05 | 3742.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 3760.00 | 3773.64 | 3743.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 3846.30 | 3773.64 | 3743.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 3722.70 | 3763.45 | 3741.85 | SL hit (close<static) qty=1.00 sl=3742.50 alert=retest2 |

### Cycle 134 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 3670.20 | 3729.12 | 3732.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 3519.00 | 3680.04 | 3709.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3560.00 | 3529.01 | 3597.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 3506.70 | 3515.97 | 3585.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 3501.10 | 3506.74 | 3563.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 3694.40 | 3548.93 | 3568.80 | SL hit (close>static) qty=1.00 sl=3627.70 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 3691.40 | 3595.19 | 3587.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 3720.60 | 3620.28 | 3599.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 3727.30 | 3730.46 | 3691.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 3859.80 | 3685.63 | 3682.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 11:15:00 | 4245.78 | 3815.17 | 3749.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 14:15:00 | 3724.70 | 3757.03 | 3760.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 10:15:00 | 3686.90 | 3732.95 | 3747.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3723.40 | 3698.42 | 3719.61 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 12:15:00 | 3793.90 | 3740.79 | 3735.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 3879.80 | 3766.44 | 3748.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 4097.30 | 4099.18 | 4020.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 4097.30 | 4099.18 | 4020.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 4067.80 | 4116.33 | 4079.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 4067.80 | 4116.33 | 4079.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 4063.40 | 4105.74 | 4077.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:15:00 | 4060.30 | 4105.74 | 4077.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4065.00 | 4097.59 | 4076.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 4060.00 | 4097.59 | 4076.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 4071.20 | 4084.30 | 4074.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 4071.20 | 4084.30 | 4074.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 4070.00 | 4081.44 | 4074.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 4135.20 | 4081.44 | 4074.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 4548.72 | 4308.09 | 4256.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 4494.70 | 4520.85 | 4521.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 4483.40 | 4508.06 | 4515.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4518.90 | 4504.90 | 4512.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 4518.90 | 4504.90 | 4512.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 4538.00 | 4511.52 | 4514.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 4550.10 | 4511.52 | 4514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 4529.10 | 4514.78 | 4515.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 4543.80 | 4514.78 | 4515.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 4500.00 | 4511.82 | 4514.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 4490.70 | 4511.82 | 4514.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 4478.50 | 4485.72 | 4494.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 09:15:00 | 4575.50 | 4509.96 | 4504.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 4718.20 | 4603.35 | 4572.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4668.00 | 4707.98 | 4653.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 4668.00 | 4707.98 | 4653.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 4633.80 | 4693.14 | 4651.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 4633.80 | 4693.14 | 4651.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 4630.00 | 4680.51 | 4649.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:45:00 | 4636.00 | 4680.51 | 4649.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 4638.60 | 4672.13 | 4648.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 4634.90 | 4672.13 | 4648.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-06-18 09:15:00 | 1624.15 | 2024-06-18 15:15:00 | 1560.00 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2024-06-19 09:15:00 | 1590.00 | 2024-06-19 14:15:00 | 1550.00 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-06-21 09:45:00 | 1596.30 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-06-21 11:15:00 | 1596.00 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-06-24 10:00:00 | 1597.85 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-06-24 11:15:00 | 1595.00 | 2024-06-24 13:15:00 | 1561.40 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-06-27 14:30:00 | 1524.95 | 2024-06-28 15:15:00 | 1448.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-27 14:30:00 | 1524.95 | 2024-06-28 15:15:00 | 1519.00 | STOP_HIT | 0.50 | 0.39% |
| SELL | retest2 | 2024-06-28 09:15:00 | 1501.85 | 2024-07-01 10:15:00 | 1566.20 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-07-01 10:30:00 | 1525.55 | 2024-07-01 11:15:00 | 1566.25 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-07-05 11:00:00 | 1685.00 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-07-05 12:00:00 | 1685.60 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-08 10:00:00 | 1680.00 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-10 09:15:00 | 1693.40 | 2024-07-10 09:15:00 | 1662.85 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-07-24 14:00:00 | 1445.10 | 2024-07-25 10:15:00 | 1525.60 | STOP_HIT | 1.00 | -5.57% |
| SELL | retest2 | 2024-07-24 15:15:00 | 1445.60 | 2024-07-25 10:15:00 | 1525.60 | STOP_HIT | 1.00 | -5.53% |
| BUY | retest2 | 2024-08-08 11:15:00 | 1767.00 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2024-08-09 11:30:00 | 1759.45 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2024-08-09 13:45:00 | 1769.00 | 2024-08-12 12:15:00 | 1713.65 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-08-26 13:15:00 | 1690.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-08-27 10:30:00 | 1695.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-08-28 10:30:00 | 1695.00 | 2024-08-28 14:15:00 | 1699.85 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1630.00 | 2024-09-06 14:15:00 | 1548.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1630.00 | 2024-09-09 14:15:00 | 1625.00 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2024-09-10 10:45:00 | 1637.80 | 2024-09-10 11:15:00 | 1665.85 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-09-19 09:15:00 | 1606.85 | 2024-09-20 09:15:00 | 1526.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 09:15:00 | 1606.85 | 2024-09-25 09:15:00 | 1553.55 | STOP_HIT | 0.50 | 3.32% |
| BUY | retest2 | 2024-09-26 12:00:00 | 1615.35 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-09-26 13:30:00 | 1613.95 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2024-09-27 10:15:00 | 1626.00 | 2024-10-03 13:15:00 | 1629.95 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest1 | 2024-10-11 09:15:00 | 1786.55 | 2024-10-15 14:15:00 | 1875.88 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-10-11 09:15:00 | 1786.55 | 2024-10-17 09:15:00 | 1965.21 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-10-14 09:15:00 | 1812.00 | 2024-10-17 09:15:00 | 1958.00 | TARGET_HIT | 1.00 | 8.06% |
| BUY | retest2 | 2024-10-14 11:15:00 | 1780.00 | 2024-10-18 11:15:00 | 1813.40 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2024-10-29 09:15:00 | 1635.85 | 2024-10-29 14:15:00 | 1689.85 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-10-29 13:00:00 | 1627.00 | 2024-10-29 14:15:00 | 1689.85 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2024-11-22 11:30:00 | 1905.50 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2024-11-25 09:15:00 | 1918.00 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.93% |
| BUY | retest2 | 2024-11-25 11:45:00 | 1909.10 | 2024-11-26 09:15:00 | 1823.50 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2024-12-09 11:15:00 | 1918.90 | 2024-12-13 10:15:00 | 1909.25 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-11 11:00:00 | 1924.00 | 2024-12-13 10:15:00 | 1909.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-12-19 09:30:00 | 2062.65 | 2024-12-24 09:15:00 | 2094.00 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2024-12-19 11:45:00 | 2061.95 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest2 | 2024-12-19 12:45:00 | 2061.50 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-12-19 13:45:00 | 2060.15 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2024-12-23 13:30:00 | 2127.30 | 2024-12-24 12:15:00 | 2080.05 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1967.95 | 2025-01-14 09:15:00 | 1869.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 12:30:00 | 1970.80 | 2025-01-14 09:15:00 | 1872.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 15:15:00 | 1951.00 | 2025-01-14 09:15:00 | 1853.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1967.95 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2025-01-10 12:30:00 | 1970.80 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 4.26% |
| SELL | retest2 | 2025-01-10 15:15:00 | 1951.00 | 2025-01-14 14:15:00 | 1886.85 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-01-23 12:30:00 | 1800.65 | 2025-01-27 09:15:00 | 1710.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 1799.10 | 2025-01-27 09:15:00 | 1709.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:30:00 | 1800.65 | 2025-01-28 09:15:00 | 1620.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 09:30:00 | 1799.10 | 2025-01-28 09:15:00 | 1619.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1414.95 | 2025-02-20 09:15:00 | 1344.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-17 09:15:00 | 1414.95 | 2025-02-20 10:15:00 | 1387.85 | STOP_HIT | 0.50 | 1.92% |
| BUY | retest2 | 2025-03-06 14:45:00 | 1391.40 | 2025-03-13 10:15:00 | 1427.40 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-03-07 10:15:00 | 1394.95 | 2025-03-13 10:15:00 | 1427.40 | STOP_HIT | 1.00 | 2.33% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1516.35 | 2025-03-20 09:15:00 | 1481.40 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-03-20 12:15:00 | 1511.45 | 2025-03-26 11:15:00 | 1547.00 | STOP_HIT | 1.00 | 2.35% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1409.25 | 2025-04-07 09:15:00 | 1338.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 13:30:00 | 1409.25 | 2025-04-08 15:15:00 | 1311.00 | STOP_HIT | 0.50 | 6.97% |
| BUY | retest2 | 2025-05-02 09:15:00 | 1571.50 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-02 10:30:00 | 1572.00 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-05-05 09:15:00 | 1575.70 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-05 12:30:00 | 1586.00 | 2025-05-06 13:15:00 | 1538.50 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-05-21 13:15:00 | 1823.90 | 2025-05-21 14:15:00 | 1849.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-02 09:15:00 | 2309.20 | 2025-06-06 13:15:00 | 2332.40 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2025-06-20 09:30:00 | 2392.40 | 2025-06-20 15:15:00 | 2345.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-06-26 09:15:00 | 2415.60 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-26 10:45:00 | 2412.50 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-06-26 11:15:00 | 2400.00 | 2025-06-26 11:15:00 | 2375.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-01 09:15:00 | 2349.50 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-01 12:45:00 | 2348.90 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-01 14:15:00 | 2348.70 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-02 09:15:00 | 2345.60 | 2025-07-02 11:15:00 | 2375.10 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-07-04 15:15:00 | 2345.20 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-07 10:30:00 | 2350.00 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-07-07 11:45:00 | 2349.40 | 2025-07-08 09:15:00 | 2372.20 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2327.80 | 2025-07-08 12:15:00 | 2383.90 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-10 12:15:00 | 2407.20 | 2025-07-10 13:15:00 | 2367.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-07-21 10:00:00 | 2407.90 | 2025-07-25 12:15:00 | 2440.00 | STOP_HIT | 1.00 | 1.33% |
| BUY | retest2 | 2025-08-04 15:15:00 | 2810.00 | 2025-08-07 15:15:00 | 2787.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-08-18 09:30:00 | 2911.90 | 2025-08-18 11:15:00 | 2839.70 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-20 10:30:00 | 2788.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-08-20 11:00:00 | 2785.00 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-08-20 14:30:00 | 2787.10 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-21 10:45:00 | 2789.80 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-08-26 09:15:00 | 2718.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-08-26 11:00:00 | 2709.60 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-08-29 10:45:00 | 2723.70 | 2025-08-29 13:15:00 | 2781.50 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-09-03 09:45:00 | 2809.20 | 2025-09-04 10:15:00 | 2756.10 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-05 15:00:00 | 2749.20 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-09-08 09:45:00 | 2752.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-08 10:30:00 | 2745.00 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-08 13:15:00 | 2751.90 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-08 14:45:00 | 2737.10 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-09 10:15:00 | 2730.70 | 2025-09-10 12:15:00 | 2773.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-12 13:45:00 | 2789.40 | 2025-09-19 10:15:00 | 3068.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 14:45:00 | 2794.00 | 2025-09-19 10:15:00 | 3064.60 | TARGET_HIT | 1.00 | 9.69% |
| BUY | retest2 | 2025-09-15 10:00:00 | 2786.00 | 2025-09-19 11:15:00 | 3073.40 | TARGET_HIT | 1.00 | 10.32% |
| SELL | retest2 | 2025-09-30 15:15:00 | 2944.60 | 2025-10-01 09:15:00 | 3034.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-10-13 14:00:00 | 3022.50 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-10-14 09:15:00 | 2968.20 | 2025-10-20 09:15:00 | 3004.10 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-24 11:30:00 | 2893.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-10-24 13:30:00 | 2900.50 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-10-24 14:45:00 | 2902.00 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-27 09:45:00 | 2873.20 | 2025-10-28 10:15:00 | 2958.60 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-11-03 09:15:00 | 3112.80 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-06 09:30:00 | 3073.60 | 2025-11-06 11:15:00 | 3059.80 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-11-07 13:30:00 | 3090.50 | 2025-11-10 09:15:00 | 3117.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-24 15:15:00 | 2843.00 | 2025-11-26 14:15:00 | 2998.30 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2025-12-01 11:45:00 | 2832.00 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-12-02 09:30:00 | 2841.80 | 2025-12-03 10:15:00 | 2900.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-12-11 13:30:00 | 2935.80 | 2025-12-17 11:15:00 | 2959.60 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-12-19 12:15:00 | 2880.60 | 2025-12-22 09:15:00 | 3085.30 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-16 14:15:00 | 2647.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-19 09:15:00 | 2610.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2787.30 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 7.61% |
| SELL | retest2 | 2026-01-16 09:15:00 | 2747.90 | 2026-01-21 14:15:00 | 2575.10 | STOP_HIT | 0.50 | 6.29% |
| BUY | retest2 | 2026-01-27 15:00:00 | 2733.40 | 2026-01-29 09:15:00 | 3006.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-17 15:15:00 | 3655.00 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-02-18 10:45:00 | 3665.50 | 2026-02-19 09:15:00 | 3566.50 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2026-03-20 09:15:00 | 3846.30 | 2026-03-20 09:15:00 | 3722.70 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2026-03-24 10:30:00 | 3506.70 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-03-24 13:45:00 | 3501.10 | 2026-03-25 09:15:00 | 3694.40 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2026-04-01 09:15:00 | 3859.80 | 2026-04-01 11:15:00 | 4245.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-01 15:00:00 | 3799.80 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-02 13:15:00 | 3820.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-02 14:30:00 | 3792.50 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-06 09:15:00 | 3884.00 | 2026-04-06 14:15:00 | 3724.70 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2026-04-17 09:15:00 | 4135.20 | 2026-04-23 09:15:00 | 4548.72 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 14:15:00 | 4490.70 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-05-04 15:00:00 | 4478.50 | 2026-05-05 09:15:00 | 4575.50 | STOP_HIT | 1.00 | -2.17% |
