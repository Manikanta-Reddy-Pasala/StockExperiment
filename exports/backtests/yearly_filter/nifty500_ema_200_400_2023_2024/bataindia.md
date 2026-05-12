# Bata India Ltd. (BATAINDIA)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 722.80
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 47 |
| PARTIAL | 15 |
| TARGET_HIT | 3 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 32
- **Target hits / Stop hits / Partials:** 3 / 44 / 15
- **Avg / median % per leg:** 0.71% / -0.06%
- **Sum % (uncompounded):** 44.21%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.96% | -31.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 0 | 0.0% | 0 | 16 | 0 | -1.96% | -31.4% |
| SELL (all) | 46 | 30 | 65.2% | 3 | 28 | 15 | 1.64% | 75.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 30 | 65.2% | 3 | 28 | 15 | 1.64% | 75.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 62 | 30 | 48.4% | 3 | 44 | 15 | 0.71% | 44.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-16 09:15:00 | 1543.00 | 1476.13 | 1475.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 09:15:00 | 1553.40 | 1487.43 | 1481.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 15:15:00 | 1699.00 | 1699.40 | 1652.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 09:15:00 | 1658.80 | 1699.40 | 1652.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 1672.50 | 1699.13 | 1652.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-10 10:15:00 | 1690.40 | 1699.13 | 1652.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 15:15:00 | 1644.40 | 1696.78 | 1653.16 | SL hit (close<static) qty=1.00 sl=1647.30 alert=retest2 |

### Cycle 2 — SELL (started 2023-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-29 15:15:00 | 1610.30 | 1665.62 | 1665.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 10:15:00 | 1605.00 | 1657.82 | 1661.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 09:15:00 | 1654.60 | 1650.55 | 1657.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-11 09:45:00 | 1650.00 | 1650.55 | 1657.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1662.50 | 1650.67 | 1657.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:00:00 | 1662.50 | 1650.67 | 1657.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 1665.90 | 1650.82 | 1657.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 11:45:00 | 1665.95 | 1650.82 | 1657.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 1609.80 | 1591.18 | 1611.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 12:00:00 | 1592.70 | 1593.88 | 1611.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 13:00:00 | 1590.25 | 1593.84 | 1610.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 10:15:00 | 1622.90 | 1594.71 | 1611.01 | SL hit (close>static) qty=1.00 sl=1622.75 alert=retest2 |

### Cycle 3 — BUY (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 15:15:00 | 1659.95 | 1620.78 | 1620.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 1660.90 | 1623.10 | 1621.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 15:15:00 | 1634.00 | 1634.50 | 1628.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 15:15:00 | 1634.00 | 1634.50 | 1628.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 15:15:00 | 1634.00 | 1634.50 | 1628.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-21 09:15:00 | 1610.00 | 1634.50 | 1628.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 1628.30 | 1634.43 | 1628.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-22 11:30:00 | 1634.40 | 1633.43 | 1627.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 12:45:00 | 1635.50 | 1631.89 | 1627.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-28 10:30:00 | 1636.40 | 1632.14 | 1627.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 11:15:00 | 1602.65 | 1633.81 | 1629.13 | SL hit (close<static) qty=1.00 sl=1602.70 alert=retest2 |

### Cycle 4 — SELL (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 11:15:00 | 1583.00 | 1625.14 | 1625.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 12:15:00 | 1575.45 | 1624.65 | 1624.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 11:15:00 | 1461.00 | 1447.82 | 1491.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-05 12:00:00 | 1461.00 | 1447.82 | 1491.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 1374.00 | 1346.62 | 1377.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 1369.70 | 1346.62 | 1377.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1379.35 | 1346.94 | 1377.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 12:00:00 | 1367.45 | 1347.40 | 1377.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 10:15:00 | 1367.70 | 1348.17 | 1377.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-22 14:00:00 | 1365.45 | 1348.73 | 1376.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 1363.35 | 1350.39 | 1373.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1372.40 | 1351.31 | 1373.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:00:00 | 1372.40 | 1351.31 | 1373.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1376.60 | 1351.56 | 1373.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1381.05 | 1351.56 | 1373.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1396.00 | 1352.00 | 1373.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 1396.00 | 1352.00 | 1373.79 | SL hit (close>static) qty=1.00 sl=1386.80 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 15:15:00 | 1475.05 | 1387.63 | 1387.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 09:15:00 | 1480.85 | 1388.56 | 1387.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-06 10:15:00 | 1540.40 | 1545.12 | 1499.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-06 11:00:00 | 1540.40 | 1545.12 | 1499.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1446.95 | 1543.13 | 1499.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:45:00 | 1449.05 | 1543.13 | 1499.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 1436.05 | 1472.89 | 1473.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1422.80 | 1467.10 | 1469.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 1446.30 | 1441.79 | 1452.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1446.30 | 1441.79 | 1452.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1446.30 | 1441.79 | 1452.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 1450.15 | 1441.79 | 1452.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1449.00 | 1441.86 | 1452.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 12:45:00 | 1441.90 | 1441.89 | 1452.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:00:00 | 1442.35 | 1441.89 | 1452.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 1442.40 | 1442.28 | 1452.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 13:00:00 | 1443.05 | 1442.29 | 1452.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 1369.81 | 1438.42 | 1449.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 1370.23 | 1438.42 | 1449.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 1370.28 | 1438.42 | 1449.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 14:15:00 | 1370.90 | 1438.42 | 1449.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 1424.30 | 1420.12 | 1436.60 | SL hit (close>ema200) qty=0.50 sl=1420.12 alert=retest2 |

### Cycle 7 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1434.50 | 1399.03 | 1398.88 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 13:15:00 | 1363.65 | 1398.55 | 1398.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 1360.70 | 1396.60 | 1397.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 1408.40 | 1386.21 | 1392.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 10:15:00 | 1408.40 | 1386.21 | 1392.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1408.40 | 1386.21 | 1392.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 1408.40 | 1386.21 | 1392.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1409.75 | 1386.44 | 1392.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:45:00 | 1390.25 | 1386.44 | 1392.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 1426.00 | 1387.13 | 1391.62 | SL hit (close>static) qty=1.00 sl=1422.90 alert=retest2 |

### Cycle 9 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 1429.90 | 1395.76 | 1395.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-10 12:15:00 | 1438.05 | 1400.37 | 1398.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 09:15:00 | 1388.35 | 1401.06 | 1398.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 09:15:00 | 1388.35 | 1401.06 | 1398.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1388.35 | 1401.06 | 1398.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 1388.35 | 1401.06 | 1398.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1372.30 | 1400.78 | 1398.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 1372.30 | 1400.78 | 1398.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 14:15:00 | 1342.00 | 1395.97 | 1396.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 1319.05 | 1394.73 | 1395.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 1340.40 | 1326.91 | 1353.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-01 13:00:00 | 1340.40 | 1326.91 | 1353.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1358.95 | 1327.22 | 1354.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:45:00 | 1364.05 | 1327.22 | 1354.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1375.00 | 1327.70 | 1354.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 12:30:00 | 1354.90 | 1334.77 | 1356.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:15:00 | 1354.10 | 1334.77 | 1356.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 13:45:00 | 1355.30 | 1339.46 | 1357.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 1348.10 | 1339.80 | 1357.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1344.00 | 1339.84 | 1357.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 1337.15 | 1339.84 | 1357.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:00:00 | 1340.00 | 1339.87 | 1356.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:00:00 | 1340.15 | 1339.67 | 1356.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 1341.00 | 1339.68 | 1356.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1345.00 | 1339.75 | 1355.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 13:00:00 | 1327.70 | 1339.63 | 1355.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-12 11:15:00 | 1377.55 | 1340.26 | 1355.37 | SL hit (close>static) qty=1.00 sl=1360.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 1263.80 | 1195.67 | 1195.34 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1164.70 | 1198.00 | 1198.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1158.90 | 1197.61 | 1197.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 932.45 | 891.36 | 936.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 932.45 | 891.36 | 936.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 932.45 | 891.36 | 936.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 889.70 | 893.82 | 935.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-13 15:15:00 | 845.22 | 891.04 | 930.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 800.73 | 868.45 | 910.86 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-10 10:15:00 | 1690.40 | 2023-08-10 15:15:00 | 1644.40 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2023-08-17 11:45:00 | 1691.15 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2023-09-01 10:15:00 | 1688.10 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2023-09-01 10:45:00 | 1687.90 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-09-05 14:45:00 | 1679.50 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2023-09-13 10:15:00 | 1676.00 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-09-13 11:00:00 | 1677.70 | 2023-09-14 13:15:00 | 1649.35 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-09-13 14:15:00 | 1677.50 | 2023-09-21 14:15:00 | 1656.05 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2023-09-14 09:15:00 | 1679.15 | 2023-09-21 14:15:00 | 1656.05 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2023-09-14 13:00:00 | 1676.10 | 2023-09-22 09:15:00 | 1646.15 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2023-09-15 10:00:00 | 1682.80 | 2023-09-22 09:15:00 | 1646.15 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2023-09-15 14:15:00 | 1676.95 | 2023-09-22 09:15:00 | 1646.15 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-11-24 12:00:00 | 1592.70 | 2023-11-28 10:15:00 | 1622.90 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2023-11-24 13:00:00 | 1590.25 | 2023-11-28 10:15:00 | 1622.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2023-12-22 11:30:00 | 1634.40 | 2024-01-02 11:15:00 | 1602.65 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2023-12-27 12:45:00 | 1635.50 | 2024-01-02 11:15:00 | 1602.65 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2023-12-28 10:30:00 | 1636.40 | 2024-01-02 11:15:00 | 1602.65 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-01-05 12:30:00 | 1634.10 | 2024-01-08 11:15:00 | 1599.95 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-05-21 12:00:00 | 1367.45 | 2024-05-30 09:15:00 | 1396.00 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-05-22 10:15:00 | 1367.70 | 2024-05-30 09:15:00 | 1396.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-05-22 14:00:00 | 1365.45 | 2024-05-30 09:15:00 | 1396.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-05-29 10:15:00 | 1363.35 | 2024-05-30 09:15:00 | 1396.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-05-31 15:15:00 | 1360.85 | 2024-06-04 12:15:00 | 1292.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 15:15:00 | 1360.85 | 2024-06-05 09:15:00 | 1410.05 | STOP_HIT | 0.50 | -3.62% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1326.00 | 2024-06-05 09:15:00 | 1410.05 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2024-09-27 12:45:00 | 1441.90 | 2024-10-03 14:15:00 | 1369.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1442.35 | 2024-10-03 14:15:00 | 1370.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 12:15:00 | 1442.40 | 2024-10-03 14:15:00 | 1370.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 13:00:00 | 1443.05 | 2024-10-03 14:15:00 | 1370.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-27 12:45:00 | 1441.90 | 2024-10-15 10:15:00 | 1424.30 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2024-09-27 14:00:00 | 1442.35 | 2024-10-15 10:15:00 | 1424.30 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2024-09-30 12:15:00 | 1442.40 | 2024-10-15 10:15:00 | 1424.30 | STOP_HIT | 0.50 | 1.25% |
| SELL | retest2 | 2024-09-30 13:00:00 | 1443.05 | 2024-10-15 10:15:00 | 1424.30 | STOP_HIT | 0.50 | 1.30% |
| SELL | retest2 | 2024-10-16 09:15:00 | 1443.20 | 2024-10-18 11:15:00 | 1455.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1444.45 | 2024-10-24 11:15:00 | 1372.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 10:15:00 | 1444.45 | 2024-11-18 09:15:00 | 1300.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 14:45:00 | 1447.00 | 2024-12-16 14:15:00 | 1434.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-12-09 09:30:00 | 1446.95 | 2024-12-16 14:15:00 | 1434.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-12-27 12:45:00 | 1390.25 | 2025-01-02 13:15:00 | 1426.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-02-04 12:30:00 | 1354.90 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-02-04 13:15:00 | 1354.10 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-02-06 13:45:00 | 1355.30 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-02-07 09:15:00 | 1348.10 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-02-07 10:15:00 | 1337.15 | 2025-02-12 11:15:00 | 1377.55 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-02-07 12:00:00 | 1340.00 | 2025-02-17 10:15:00 | 1287.15 | PARTIAL | 0.50 | 3.94% |
| SELL | retest2 | 2025-02-10 10:00:00 | 1340.15 | 2025-02-17 10:15:00 | 1286.39 | PARTIAL | 0.50 | 4.01% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1341.00 | 2025-02-17 10:15:00 | 1287.53 | PARTIAL | 0.50 | 3.99% |
| SELL | retest2 | 2025-02-11 13:00:00 | 1327.70 | 2025-02-17 11:15:00 | 1280.69 | PARTIAL | 0.50 | 3.54% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1321.55 | 2025-02-19 12:15:00 | 1259.94 | PARTIAL | 0.50 | 4.66% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1326.25 | 2025-02-19 13:15:00 | 1255.47 | PARTIAL | 0.50 | 5.34% |
| SELL | retest2 | 2025-02-14 09:15:00 | 1322.95 | 2025-02-19 13:15:00 | 1256.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:00:00 | 1340.00 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 1.32% |
| SELL | retest2 | 2025-02-10 10:00:00 | 1340.15 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 1.33% |
| SELL | retest2 | 2025-02-10 11:00:00 | 1341.00 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-02-11 13:00:00 | 1327.70 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 0.40% |
| SELL | retest2 | 2025-02-13 09:15:00 | 1321.55 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | -0.06% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1326.25 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 0.29% |
| SELL | retest2 | 2025-02-14 09:15:00 | 1322.95 | 2025-02-24 10:15:00 | 1322.35 | STOP_HIT | 0.50 | 0.05% |
| SELL | retest2 | 2025-02-27 10:45:00 | 1312.30 | 2025-02-28 12:15:00 | 1246.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 10:45:00 | 1312.30 | 2025-03-03 10:15:00 | 1181.07 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 889.70 | 2026-02-13 15:15:00 | 845.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 09:15:00 | 889.70 | 2026-02-24 09:15:00 | 800.73 | TARGET_HIT | 0.50 | 10.00% |
