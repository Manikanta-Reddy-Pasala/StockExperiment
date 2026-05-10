# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 64 |
| ALERT1 | 46 |
| ALERT2 | 46 |
| ALERT2_SKIP | 25 |
| ALERT3 | 140 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 65 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 44
- **Target hits / Stop hits / Partials:** 0 / 68 / 11
- **Avg / median % per leg:** 0.49% / -0.62%
- **Sum % (uncompounded):** 39.05%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 0 | 31 | 3 | 0.21% | 7.1% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.61% | 27.7% |
| BUY @ 3rd Alert (retest2) | 28 | 8 | 28.6% | 0 | 28 | 0 | -0.74% | -20.6% |
| SELL (all) | 45 | 21 | 46.7% | 0 | 37 | 8 | 0.71% | 32.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 21 | 46.7% | 0 | 37 | 8 | 0.71% | 32.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.61% | 27.7% |
| retest2 (combined) | 73 | 29 | 39.7% | 0 | 65 | 8 | 0.16% | 11.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1293.00 | 1255.94 | 1255.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1296.10 | 1263.97 | 1259.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 1298.00 | 1300.50 | 1287.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 1316.60 | 1304.22 | 1290.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 09:15:00 | 1382.43 | 1351.43 | 1329.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1389.00 | 1372.68 | 1353.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 1362.80 | 1373.48 | 1360.30 | SL hit (close<ema200) qty=0.50 sl=1373.48 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:30:00 | 1396.70 | 1374.13 | 1364.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 09:15:00 | 1355.00 | 1361.19 | 1361.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 1355.00 | 1361.19 | 1361.71 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1383.80 | 1363.02 | 1361.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1390.50 | 1368.52 | 1364.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1362.20 | 1369.49 | 1365.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 1362.20 | 1369.49 | 1365.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1361.00 | 1367.79 | 1365.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:45:00 | 1360.20 | 1367.79 | 1365.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1370.00 | 1368.24 | 1365.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:15:00 | 1375.00 | 1367.61 | 1365.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1370.10 | 1369.37 | 1366.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:30:00 | 1371.00 | 1371.08 | 1368.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 1356.30 | 1370.37 | 1369.87 | SL hit (close<static) qty=1.00 sl=1358.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 1356.30 | 1370.37 | 1369.87 | SL hit (close<static) qty=1.00 sl=1358.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 1356.30 | 1370.37 | 1369.87 | SL hit (close<static) qty=1.00 sl=1358.50 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1355.20 | 1367.33 | 1368.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1344.20 | 1360.97 | 1365.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1365.00 | 1360.66 | 1364.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 1364.40 | 1360.66 | 1364.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1350.50 | 1358.63 | 1363.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:15:00 | 1348.20 | 1358.63 | 1363.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 1348.80 | 1355.58 | 1360.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1387.10 | 1361.63 | 1362.69 | SL hit (close>static) qty=1.00 sl=1367.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 1387.10 | 1361.63 | 1362.69 | SL hit (close>static) qty=1.00 sl=1367.30 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 1395.30 | 1368.36 | 1365.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1425.00 | 1394.26 | 1384.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 15:15:00 | 1420.00 | 1422.93 | 1405.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1443.10 | 1422.93 | 1405.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-02 11:45:00 | 1446.40 | 1434.11 | 1415.69 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 1515.25 | 1478.77 | 1448.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:15:00 | 1518.72 | 1478.77 | 1448.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 1511.00 | 1513.46 | 1483.07 | SL hit (close<ema200) qty=0.50 sl=1513.46 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-04 10:15:00 | 1511.00 | 1513.46 | 1483.07 | SL hit (close<ema200) qty=0.50 sl=1513.46 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1619.80 | 1652.16 | 1639.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 1619.80 | 1652.16 | 1639.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1615.10 | 1644.75 | 1636.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 1615.40 | 1644.75 | 1636.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1635.40 | 1645.51 | 1639.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1638.40 | 1645.51 | 1639.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1632.30 | 1642.87 | 1638.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:45:00 | 1625.20 | 1642.87 | 1638.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1628.40 | 1639.97 | 1637.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 1628.40 | 1639.97 | 1637.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 1619.90 | 1635.96 | 1636.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 13:15:00 | 1610.80 | 1630.93 | 1633.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1581.00 | 1565.57 | 1578.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1581.00 | 1565.57 | 1578.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1589.20 | 1570.30 | 1579.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1589.20 | 1570.30 | 1579.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1584.00 | 1573.04 | 1579.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 1591.20 | 1573.04 | 1579.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1600.00 | 1580.10 | 1581.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1587.40 | 1580.10 | 1581.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 1603.00 | 1584.68 | 1583.89 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1567.80 | 1586.59 | 1587.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1556.20 | 1577.94 | 1583.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1547.00 | 1546.98 | 1560.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:15:00 | 1552.60 | 1546.98 | 1560.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1521.50 | 1508.75 | 1521.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:30:00 | 1524.60 | 1508.75 | 1521.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1518.50 | 1510.70 | 1521.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:15:00 | 1520.80 | 1510.70 | 1521.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 1520.00 | 1512.56 | 1521.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 12:30:00 | 1512.60 | 1513.77 | 1520.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:15:00 | 1516.40 | 1513.77 | 1520.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 10:00:00 | 1512.00 | 1509.42 | 1516.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 1509.90 | 1513.16 | 1515.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1464.40 | 1464.92 | 1473.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 1491.00 | 1479.52 | 1478.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1506.30 | 1490.41 | 1485.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1508.10 | 1510.93 | 1502.74 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1500.00 | 1511.95 | 1511.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 1496.20 | 1508.80 | 1510.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 11:15:00 | 1537.00 | 1509.01 | 1509.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 12:00:00 | 1537.00 | 1509.01 | 1509.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1527.40 | 1512.69 | 1510.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1583.00 | 1539.20 | 1524.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 1685.10 | 1688.15 | 1668.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 1695.80 | 1688.15 | 1668.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1674.00 | 1683.60 | 1670.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 1673.00 | 1683.60 | 1670.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1668.40 | 1686.36 | 1682.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 1668.40 | 1686.36 | 1682.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1671.00 | 1683.29 | 1681.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1642.10 | 1683.29 | 1681.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1643.70 | 1675.37 | 1677.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 10:15:00 | 1626.00 | 1645.83 | 1658.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 1615.00 | 1613.51 | 1628.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 15:00:00 | 1615.00 | 1613.51 | 1628.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1599.50 | 1611.59 | 1625.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 1580.50 | 1597.38 | 1611.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 1580.20 | 1574.58 | 1582.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1560.50 | 1576.79 | 1582.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 1602.10 | 1582.17 | 1582.17 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 1566.50 | 1582.07 | 1583.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 11:15:00 | 1552.50 | 1572.44 | 1578.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 1578.00 | 1573.55 | 1578.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 1578.00 | 1573.55 | 1578.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1581.40 | 1575.12 | 1579.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1581.40 | 1575.12 | 1579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1566.30 | 1573.36 | 1577.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1582.40 | 1573.36 | 1577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1562.80 | 1569.91 | 1575.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:30:00 | 1578.10 | 1569.91 | 1575.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1587.00 | 1573.39 | 1576.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1587.00 | 1573.39 | 1576.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1587.90 | 1576.29 | 1577.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1587.90 | 1576.29 | 1577.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 13:15:00 | 1586.10 | 1578.25 | 1577.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1610.00 | 1586.07 | 1581.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1579.90 | 1584.83 | 1581.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1579.90 | 1584.83 | 1581.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1577.50 | 1583.37 | 1581.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 1576.20 | 1583.37 | 1581.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1575.00 | 1581.69 | 1580.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:45:00 | 1580.10 | 1581.69 | 1580.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 1575.60 | 1580.47 | 1580.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 1573.40 | 1580.47 | 1580.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 1578.70 | 1580.12 | 1580.00 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 1560.00 | 1576.10 | 1578.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 1555.10 | 1571.90 | 1576.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1555.60 | 1552.17 | 1561.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 14:45:00 | 1553.80 | 1552.17 | 1561.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1556.10 | 1552.95 | 1561.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1535.00 | 1552.95 | 1561.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 10:15:00 | 1521.40 | 1517.53 | 1517.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1521.40 | 1517.53 | 1517.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 1539.30 | 1521.88 | 1519.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 1521.80 | 1526.34 | 1522.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 1514.90 | 1526.34 | 1522.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1527.90 | 1526.65 | 1523.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 1514.90 | 1526.65 | 1523.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1514.10 | 1524.14 | 1522.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:45:00 | 1512.00 | 1524.14 | 1522.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1508.30 | 1520.97 | 1521.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 12:15:00 | 1505.60 | 1517.90 | 1519.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 1501.00 | 1499.07 | 1504.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 1501.00 | 1499.07 | 1504.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1503.70 | 1500.00 | 1503.97 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1523.00 | 1506.58 | 1506.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1534.40 | 1512.15 | 1508.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 14:15:00 | 1517.60 | 1519.60 | 1514.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 15:00:00 | 1517.60 | 1519.60 | 1514.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1525.20 | 1521.23 | 1515.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 1531.90 | 1521.23 | 1515.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1508.90 | 1518.57 | 1515.59 | SL hit (close<static) qty=1.00 sl=1511.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1494.10 | 1510.70 | 1512.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1474.60 | 1503.48 | 1508.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1500.00 | 1497.24 | 1504.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1500.00 | 1497.24 | 1504.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1504.30 | 1498.65 | 1504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1504.30 | 1498.65 | 1504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1505.00 | 1499.92 | 1504.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1483.00 | 1502.20 | 1504.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 13:15:00 | 1510.40 | 1497.19 | 1500.02 | SL hit (close>static) qty=1.00 sl=1508.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:00:00 | 1479.20 | 1493.59 | 1498.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 14:15:00 | 1405.24 | 1425.72 | 1437.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1428.30 | 1424.68 | 1435.20 | SL hit (close>ema200) qty=0.50 sl=1424.68 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1452.00 | 1435.47 | 1434.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1463.20 | 1443.68 | 1438.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 1593.90 | 1594.76 | 1570.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 1593.90 | 1594.76 | 1570.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1612.00 | 1623.66 | 1612.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:30:00 | 1611.90 | 1623.66 | 1612.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1609.90 | 1620.91 | 1612.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 11:30:00 | 1611.60 | 1620.91 | 1612.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1607.50 | 1618.23 | 1612.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:30:00 | 1606.70 | 1618.23 | 1612.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1603.90 | 1610.36 | 1609.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1603.20 | 1610.36 | 1609.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1610.60 | 1610.38 | 1609.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1610.00 | 1610.38 | 1609.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1603.10 | 1608.92 | 1609.22 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 12:15:00 | 1613.80 | 1609.90 | 1609.64 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 1598.30 | 1607.58 | 1608.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1592.00 | 1604.46 | 1607.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1578.60 | 1574.20 | 1580.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 1558.00 | 1573.89 | 1578.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1565.20 | 1573.89 | 1578.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 14:00:00 | 1563.20 | 1571.76 | 1577.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1549.60 | 1568.66 | 1574.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1537.30 | 1520.18 | 1532.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1537.30 | 1520.18 | 1532.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 1553.40 | 1526.82 | 1534.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1532.50 | 1526.82 | 1534.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1480.10 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1486.94 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 12:15:00 | 1485.04 | 1504.36 | 1516.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-03 13:15:00 | 1472.12 | 1498.51 | 1512.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1473.50 | 1473.24 | 1484.86 | SL hit (close>ema200) qty=0.50 sl=1473.24 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1473.50 | 1473.24 | 1484.86 | SL hit (close>ema200) qty=0.50 sl=1473.24 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1473.50 | 1473.24 | 1484.86 | SL hit (close>ema200) qty=0.50 sl=1473.24 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1473.50 | 1473.24 | 1484.86 | SL hit (close>ema200) qty=0.50 sl=1473.24 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 15:15:00 | 1455.88 | 1465.58 | 1473.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1468.70 | 1466.21 | 1472.87 | SL hit (close>ema200) qty=0.50 sl=1466.21 alert=retest2 |

### Cycle 25 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1502.00 | 1447.14 | 1445.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 1537.20 | 1465.15 | 1453.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1527.50 | 1527.86 | 1506.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 15:15:00 | 1519.80 | 1526.39 | 1515.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1519.80 | 1526.39 | 1515.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:30:00 | 1547.80 | 1528.57 | 1517.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 11:15:00 | 1503.50 | 1521.22 | 1515.56 | SL hit (close<static) qty=1.00 sl=1510.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 1533.90 | 1523.60 | 1517.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 1508.40 | 1529.80 | 1525.71 | SL hit (close<static) qty=1.00 sl=1510.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 11:15:00 | 1530.00 | 1529.80 | 1525.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1535.90 | 1532.08 | 1529.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1536.10 | 1534.22 | 1531.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1552.00 | 1534.34 | 1531.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1533.00 | 1550.11 | 1550.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1533.00 | 1550.11 | 1550.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 1533.00 | 1550.11 | 1550.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 1533.00 | 1550.11 | 1550.22 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1558.30 | 1551.75 | 1550.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1565.00 | 1557.57 | 1554.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1552.20 | 1556.50 | 1554.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1552.20 | 1556.50 | 1554.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1555.20 | 1556.24 | 1554.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 1552.90 | 1556.24 | 1554.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1572.00 | 1559.39 | 1555.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 1578.40 | 1559.39 | 1555.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:30:00 | 1576.50 | 1565.19 | 1559.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:15:00 | 1576.10 | 1567.15 | 1560.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1615.80 | 1649.58 | 1654.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 14:15:00 | 1610.10 | 1630.04 | 1642.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1624.10 | 1618.20 | 1627.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1631.50 | 1618.20 | 1627.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1626.80 | 1619.92 | 1627.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:30:00 | 1628.70 | 1619.92 | 1627.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1625.00 | 1620.94 | 1626.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1624.10 | 1620.94 | 1626.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1604.00 | 1592.03 | 1602.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 1604.00 | 1592.03 | 1602.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1607.10 | 1595.05 | 1602.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:45:00 | 1609.90 | 1595.05 | 1602.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1607.00 | 1597.44 | 1602.94 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 1619.50 | 1608.02 | 1606.68 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1586.70 | 1603.76 | 1604.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 1577.60 | 1598.52 | 1602.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 13:15:00 | 1569.50 | 1567.76 | 1575.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 13:45:00 | 1566.50 | 1567.76 | 1575.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1557.30 | 1565.66 | 1573.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 14:30:00 | 1570.50 | 1565.66 | 1573.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1536.70 | 1556.73 | 1567.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:45:00 | 1531.50 | 1551.96 | 1564.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 14:45:00 | 1523.50 | 1540.70 | 1554.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:45:00 | 1526.00 | 1525.27 | 1533.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 1545.10 | 1538.46 | 1537.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1559.50 | 1542.67 | 1539.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1548.30 | 1559.00 | 1552.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 1551.90 | 1559.00 | 1552.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1544.40 | 1556.08 | 1551.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 1543.00 | 1556.08 | 1551.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1545.10 | 1553.88 | 1551.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:45:00 | 1545.10 | 1553.88 | 1551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1535.60 | 1547.72 | 1548.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1532.30 | 1542.65 | 1546.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 1543.60 | 1539.31 | 1543.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 1543.60 | 1539.31 | 1543.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 1535.50 | 1538.55 | 1542.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:45:00 | 1542.00 | 1538.55 | 1542.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 1534.00 | 1537.64 | 1541.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 1544.00 | 1537.64 | 1541.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1549.70 | 1540.05 | 1542.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:15:00 | 1537.30 | 1540.82 | 1542.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1528.20 | 1539.13 | 1541.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1555.00 | 1540.80 | 1539.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-02 15:15:00 | 1555.00 | 1540.80 | 1539.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 1555.00 | 1540.80 | 1539.87 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1530.40 | 1538.72 | 1539.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1524.00 | 1535.78 | 1537.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1534.50 | 1528.78 | 1532.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1534.50 | 1528.78 | 1532.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1532.60 | 1529.54 | 1532.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:15:00 | 1535.20 | 1529.54 | 1532.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1536.30 | 1530.89 | 1532.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:45:00 | 1542.80 | 1530.89 | 1532.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1532.00 | 1531.11 | 1532.65 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 15:15:00 | 1543.00 | 1534.90 | 1534.19 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1505.40 | 1529.14 | 1532.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1496.40 | 1518.73 | 1526.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1461.00 | 1444.99 | 1468.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 09:30:00 | 1453.30 | 1444.99 | 1468.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1430.80 | 1421.51 | 1436.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 1430.10 | 1421.51 | 1436.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1440.00 | 1425.76 | 1435.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:30:00 | 1433.80 | 1425.76 | 1435.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1438.60 | 1428.33 | 1436.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1441.90 | 1428.33 | 1436.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 1437.00 | 1433.72 | 1435.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 1437.00 | 1433.72 | 1435.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 1430.00 | 1432.97 | 1435.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1416.10 | 1432.97 | 1435.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 10:15:00 | 1440.00 | 1432.94 | 1434.81 | SL hit (close>static) qty=1.00 sl=1438.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 12:15:00 | 1445.00 | 1437.17 | 1436.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1458.40 | 1441.42 | 1438.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1445.10 | 1449.95 | 1443.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 1445.10 | 1449.95 | 1443.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1435.00 | 1446.96 | 1443.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 15:15:00 | 1451.00 | 1443.34 | 1442.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1432.90 | 1442.48 | 1442.12 | SL hit (close<static) qty=1.00 sl=1433.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 1449.90 | 1442.98 | 1442.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1456.40 | 1444.32 | 1443.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 1452.00 | 1445.61 | 1444.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1481.10 | 1487.05 | 1479.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:30:00 | 1479.10 | 1487.05 | 1479.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1473.10 | 1484.26 | 1479.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 13:00:00 | 1473.10 | 1484.26 | 1479.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1476.50 | 1482.71 | 1478.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:30:00 | 1481.90 | 1478.13 | 1477.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 14:45:00 | 1481.80 | 1477.86 | 1477.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 09:15:00 | 1473.30 | 1477.23 | 1477.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 1467.10 | 1474.40 | 1476.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 11:15:00 | 1461.00 | 1459.10 | 1466.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 11:45:00 | 1461.00 | 1459.10 | 1466.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 1469.40 | 1459.86 | 1465.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 1464.40 | 1459.86 | 1465.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1457.20 | 1459.33 | 1464.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 1468.60 | 1459.33 | 1464.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1466.70 | 1460.11 | 1463.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 1462.20 | 1460.11 | 1463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1477.50 | 1463.59 | 1465.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 1477.50 | 1463.59 | 1465.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1480.90 | 1467.05 | 1466.56 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 1459.30 | 1465.50 | 1465.90 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1474.80 | 1466.24 | 1465.29 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 15:15:00 | 1459.30 | 1464.13 | 1464.51 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1471.40 | 1465.59 | 1465.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1484.20 | 1469.31 | 1466.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 1550.70 | 1554.96 | 1526.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 1550.70 | 1554.96 | 1526.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1535.00 | 1544.99 | 1534.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 1535.00 | 1544.99 | 1534.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1534.60 | 1542.92 | 1534.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 1534.60 | 1542.92 | 1534.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1526.10 | 1539.55 | 1533.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1510.40 | 1539.55 | 1533.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1526.90 | 1537.02 | 1533.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 1525.40 | 1537.02 | 1533.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1525.70 | 1533.55 | 1532.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1526.00 | 1533.55 | 1532.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1561.80 | 1539.20 | 1534.95 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 1537.90 | 1541.69 | 1541.94 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 14:15:00 | 1547.30 | 1542.81 | 1542.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-09 15:15:00 | 1570.00 | 1548.25 | 1544.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1530.10 | 1544.62 | 1543.59 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 1531.00 | 1541.69 | 1543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 13:15:00 | 1520.70 | 1535.55 | 1539.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1534.30 | 1530.54 | 1535.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 1536.90 | 1530.54 | 1535.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1553.90 | 1535.21 | 1537.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:45:00 | 1553.60 | 1535.21 | 1537.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1546.80 | 1537.53 | 1537.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1546.80 | 1537.53 | 1537.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2026-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 14:15:00 | 1550.10 | 1540.04 | 1539.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1561.30 | 1545.89 | 1541.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 1540.20 | 1548.59 | 1544.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:45:00 | 1542.30 | 1548.59 | 1544.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1526.90 | 1544.25 | 1543.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 1526.90 | 1544.25 | 1543.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 1528.00 | 1541.00 | 1541.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 1500.60 | 1532.92 | 1538.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1366.00 | 1362.60 | 1405.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1375.70 | 1362.60 | 1405.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1369.40 | 1353.58 | 1368.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 1368.20 | 1353.58 | 1368.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1373.00 | 1357.46 | 1369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1373.00 | 1357.46 | 1369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1393.40 | 1364.65 | 1371.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1362.30 | 1364.65 | 1371.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:15:00 | 1364.60 | 1366.06 | 1371.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 1371.20 | 1362.96 | 1367.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 1366.90 | 1364.55 | 1367.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1368.20 | 1365.28 | 1367.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 1369.10 | 1365.28 | 1367.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1399.50 | 1372.13 | 1370.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1408.00 | 1383.36 | 1375.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1389.90 | 1394.79 | 1383.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 1389.90 | 1394.79 | 1383.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1384.50 | 1392.73 | 1383.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1382.40 | 1392.73 | 1383.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 1380.60 | 1390.30 | 1383.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 1380.60 | 1390.30 | 1383.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 1396.80 | 1391.60 | 1384.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 13:30:00 | 1404.90 | 1394.00 | 1386.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 1406.00 | 1401.07 | 1391.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1366.10 | 1415.21 | 1413.13 | SL hit (close<static) qty=1.00 sl=1378.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-01 14:15:00 | 1366.10 | 1415.21 | 1413.13 | SL hit (close<static) qty=1.00 sl=1378.10 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1375.00 | 1407.16 | 1409.67 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1421.50 | 1408.43 | 1408.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1484.80 | 1425.55 | 1416.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 1475.10 | 1478.69 | 1456.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:30:00 | 1469.80 | 1478.69 | 1456.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1497.80 | 1486.98 | 1477.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1507.50 | 1491.30 | 1483.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1501.20 | 1525.46 | 1526.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1501.20 | 1525.46 | 1526.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1490.40 | 1510.46 | 1517.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 12:15:00 | 1488.10 | 1487.68 | 1497.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 12:45:00 | 1490.30 | 1487.68 | 1497.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1492.90 | 1488.73 | 1496.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1492.90 | 1488.73 | 1496.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 1479.40 | 1481.38 | 1488.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 1475.10 | 1481.64 | 1486.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 1500.50 | 1483.36 | 1485.71 | SL hit (close>static) qty=1.00 sl=1488.60 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1517.20 | 1492.18 | 1488.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1527.60 | 1507.65 | 1497.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 14:15:00 | 1513.90 | 1514.66 | 1505.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 1513.90 | 1514.66 | 1505.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1497.30 | 1510.92 | 1505.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1493.40 | 1510.92 | 1505.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1497.40 | 1508.22 | 1504.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:00:00 | 1497.40 | 1508.22 | 1504.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1490.20 | 1504.61 | 1503.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 12:00:00 | 1490.20 | 1504.61 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 1479.70 | 1499.63 | 1500.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 1469.80 | 1493.67 | 1498.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1468.50 | 1458.12 | 1470.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:30:00 | 1465.40 | 1458.12 | 1470.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1347.80 | 1339.17 | 1362.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1307.50 | 1353.49 | 1355.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 1323.50 | 1315.54 | 1323.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1339.40 | 1328.45 | 1327.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 1339.40 | 1328.45 | 1327.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 1339.40 | 1328.45 | 1327.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 14:15:00 | 1354.50 | 1337.13 | 1332.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1331.40 | 1339.65 | 1334.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 1356.70 | 1342.34 | 1336.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1312.60 | 1333.80 | 1334.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1312.60 | 1333.80 | 1334.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1302.60 | 1327.56 | 1331.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1259.90 | 1257.04 | 1279.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 1259.90 | 1257.04 | 1279.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1276.00 | 1261.61 | 1274.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1276.00 | 1261.61 | 1274.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 1276.00 | 1264.49 | 1274.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 15:00:00 | 1276.00 | 1264.49 | 1274.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 1282.00 | 1267.99 | 1275.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 1295.90 | 1273.01 | 1277.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1296.30 | 1277.67 | 1278.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 1296.80 | 1277.67 | 1278.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1288.10 | 1279.76 | 1279.61 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1262.90 | 1279.56 | 1280.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1252.00 | 1269.81 | 1275.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 1266.40 | 1263.55 | 1270.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 14:15:00 | 1268.40 | 1263.55 | 1270.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1228.60 | 1215.63 | 1227.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1230.40 | 1215.63 | 1227.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1208.10 | 1214.12 | 1226.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:15:00 | 1200.20 | 1214.12 | 1226.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1248.30 | 1218.73 | 1225.93 | SL hit (close>static) qty=1.00 sl=1229.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 1201.80 | 1225.69 | 1227.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1245.00 | 1219.77 | 1220.43 | SL hit (close>static) qty=1.00 sl=1229.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 1200.30 | 1219.77 | 1220.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 10:00:00 | 1195.90 | 1214.99 | 1218.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1229.90 | 1201.57 | 1207.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1229.90 | 1201.57 | 1207.64 | SL hit (close>static) qty=1.00 sl=1229.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1229.90 | 1201.57 | 1207.64 | SL hit (close>static) qty=1.00 sl=1229.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1233.80 | 1201.57 | 1207.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1208.40 | 1202.93 | 1207.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:15:00 | 1203.10 | 1202.93 | 1207.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:45:00 | 1200.00 | 1203.92 | 1207.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1142.94 | 1188.24 | 1198.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1140.00 | 1188.24 | 1198.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1167.90 | 1165.47 | 1180.30 | SL hit (close>ema200) qty=0.50 sl=1165.47 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1167.90 | 1165.47 | 1180.30 | SL hit (close>ema200) qty=0.50 sl=1165.47 alert=retest2 |

### Cycle 59 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1222.60 | 1191.21 | 1187.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 15:15:00 | 1230.00 | 1198.97 | 1191.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1277.60 | 1282.86 | 1256.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 1276.70 | 1282.86 | 1256.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1289.20 | 1298.64 | 1289.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:30:00 | 1294.80 | 1299.33 | 1290.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 1309.00 | 1318.29 | 1319.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1309.00 | 1318.29 | 1319.16 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1369.50 | 1328.53 | 1323.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 13:15:00 | 1390.80 | 1368.60 | 1354.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 10:15:00 | 1407.20 | 1411.55 | 1392.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-24 11:00:00 | 1407.20 | 1411.55 | 1392.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 1402.10 | 1408.29 | 1394.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 13:00:00 | 1402.10 | 1408.29 | 1394.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1417.00 | 1423.54 | 1413.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:45:00 | 1418.00 | 1423.54 | 1413.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1438.10 | 1424.26 | 1416.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1426.20 | 1424.26 | 1416.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1433.20 | 1443.42 | 1434.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1456.00 | 1436.41 | 1434.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 12:30:00 | 1453.20 | 1446.29 | 1440.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1498.40 | 1443.83 | 1441.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 1457.00 | 1453.92 | 1450.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1444.00 | 1451.94 | 1449.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1444.00 | 1451.94 | 1449.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1437.50 | 1449.05 | 1448.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 1436.00 | 1449.05 | 1448.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 11:15:00 | 1436.60 | 1446.56 | 1447.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 09:15:00 | 1421.30 | 1440.13 | 1444.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 1440.00 | 1438.08 | 1442.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:00:00 | 1440.00 | 1438.08 | 1442.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 1449.00 | 1440.27 | 1442.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 1448.60 | 1440.27 | 1442.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 1449.40 | 1442.09 | 1443.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:15:00 | 1446.00 | 1442.09 | 1443.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 14:15:00 | 1455.00 | 1444.68 | 1444.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 1455.00 | 1444.68 | 1444.56 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1430.50 | 1442.69 | 1443.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 11:15:00 | 1428.20 | 1437.92 | 1441.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1442.50 | 1438.84 | 1441.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1442.50 | 1438.84 | 1441.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1435.70 | 1438.21 | 1440.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1435.70 | 1438.21 | 1440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:30:00 | 1316.60 | 2025-05-16 09:15:00 | 1382.43 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:30:00 | 1316.60 | 2025-05-19 13:15:00 | 1362.80 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-05-20 10:30:00 | 1396.70 | 2025-05-21 09:15:00 | 1355.00 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2025-05-23 09:15:00 | 1375.00 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-05-23 11:15:00 | 1370.10 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-05-23 13:30:00 | 1371.00 | 2025-05-26 13:15:00 | 1356.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-05-27 13:15:00 | 1348.20 | 2025-05-28 09:15:00 | 1387.10 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-05-27 14:45:00 | 1348.80 | 2025-05-28 09:15:00 | 1387.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest1 | 2025-06-02 09:15:00 | 1443.10 | 2025-06-03 11:15:00 | 1515.25 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-02 11:45:00 | 1446.40 | 2025-06-03 11:15:00 | 1518.72 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-02 09:15:00 | 1443.10 | 2025-06-04 10:15:00 | 1511.00 | STOP_HIT | 0.50 | 4.71% |
| BUY | retest1 | 2025-06-02 11:45:00 | 1446.40 | 2025-06-04 10:15:00 | 1511.00 | STOP_HIT | 0.50 | 4.47% |
| SELL | retest2 | 2025-06-24 12:30:00 | 1512.60 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2025-06-24 13:15:00 | 1516.40 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-06-25 10:00:00 | 1512.00 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.39% |
| SELL | retest2 | 2025-06-26 09:30:00 | 1509.90 | 2025-07-01 13:15:00 | 1491.00 | STOP_HIT | 1.00 | 1.25% |
| SELL | retest2 | 2025-07-29 09:30:00 | 1580.50 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-07-30 14:30:00 | 1580.20 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1560.50 | 2025-07-31 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-08-08 09:15:00 | 1535.00 | 2025-08-14 10:15:00 | 1521.40 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-08-22 10:15:00 | 1531.90 | 2025-08-22 11:15:00 | 1508.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1483.00 | 2025-08-26 13:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-26 15:00:00 | 1479.20 | 2025-09-05 14:15:00 | 1405.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 15:00:00 | 1479.20 | 2025-09-08 09:15:00 | 1428.30 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1558.00 | 2025-10-03 12:15:00 | 1480.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1565.20 | 2025-10-03 12:15:00 | 1486.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 14:00:00 | 1563.20 | 2025-10-03 12:15:00 | 1485.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1549.60 | 2025-10-03 13:15:00 | 1472.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 12:30:00 | 1558.00 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.42% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1565.20 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2025-09-25 14:00:00 | 1563.20 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 5.74% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1549.60 | 2025-10-07 12:15:00 | 1473.50 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1532.50 | 2025-10-08 15:15:00 | 1455.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-01 09:15:00 | 1532.50 | 2025-10-09 09:15:00 | 1468.70 | STOP_HIT | 0.50 | 4.16% |
| BUY | retest2 | 2025-10-20 09:30:00 | 1547.80 | 2025-10-20 11:15:00 | 1503.50 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-10-20 12:45:00 | 1533.90 | 2025-10-23 10:15:00 | 1508.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-23 11:15:00 | 1530.00 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1535.90 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1552.00 | 2025-10-29 09:15:00 | 1533.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-30 12:15:00 | 1578.40 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2025-10-30 13:30:00 | 1576.50 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest2 | 2025-10-30 15:15:00 | 1576.10 | 2025-11-11 10:15:00 | 1615.80 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-11-21 10:45:00 | 1531.50 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-11-21 14:45:00 | 1523.50 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-25 10:45:00 | 1526.00 | 2025-11-25 15:15:00 | 1545.10 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-01 14:15:00 | 1537.30 | 2025-12-02 15:15:00 | 1555.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1528.20 | 2025-12-02 15:15:00 | 1555.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1416.10 | 2025-12-16 10:15:00 | 1440.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-12-17 15:15:00 | 1451.00 | 2025-12-18 09:15:00 | 1432.90 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-18 11:15:00 | 1449.90 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.61% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1456.40 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-12-19 12:45:00 | 1452.00 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-12-26 13:30:00 | 1481.90 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-12-26 14:45:00 | 1481.80 | 2025-12-29 09:15:00 | 1473.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1362.30 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2026-01-27 10:15:00 | 1364.60 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2026-01-27 14:30:00 | 1371.20 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-01-28 10:00:00 | 1366.90 | 2026-01-28 11:15:00 | 1399.50 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-01-29 13:30:00 | 1404.90 | 2026-02-01 14:15:00 | 1366.10 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2026-01-30 09:30:00 | 1406.00 | 2026-02-01 14:15:00 | 1366.10 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1507.50 | 2026-02-13 09:15:00 | 1501.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-02-19 11:45:00 | 1475.10 | 2026-02-19 14:15:00 | 1500.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1307.50 | 2026-03-11 11:15:00 | 1339.40 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2026-03-10 15:00:00 | 1323.50 | 2026-03-11 11:15:00 | 1339.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-03-12 11:30:00 | 1356.70 | 2026-03-13 09:15:00 | 1312.60 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2026-03-24 15:15:00 | 1200.20 | 2026-03-25 09:15:00 | 1248.30 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2026-03-27 09:15:00 | 1201.80 | 2026-03-27 15:15:00 | 1245.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1200.30 | 2026-04-01 09:15:00 | 1229.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-03-30 10:00:00 | 1195.90 | 2026-04-01 09:15:00 | 1229.90 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1203.10 | 2026-04-02 09:15:00 | 1142.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1200.00 | 2026-04-02 09:15:00 | 1140.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 11:15:00 | 1203.10 | 2026-04-02 15:15:00 | 1167.90 | STOP_HIT | 0.50 | 2.93% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1200.00 | 2026-04-02 15:15:00 | 1167.90 | STOP_HIT | 0.50 | 2.67% |
| BUY | retest2 | 2026-04-13 10:30:00 | 1294.80 | 2026-04-20 15:15:00 | 1309.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1456.00 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-05-04 12:30:00 | 1453.20 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-05-05 09:15:00 | 1498.40 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -4.12% |
| BUY | retest2 | 2026-05-06 09:15:00 | 1457.00 | 2026-05-06 11:15:00 | 1436.60 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-07 14:15:00 | 1446.00 | 2026-05-07 14:15:00 | 1455.00 | STOP_HIT | 1.00 | -0.62% |
