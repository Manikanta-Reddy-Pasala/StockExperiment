# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1075.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 229 |
| ALERT1 | 149 |
| ALERT2 | 145 |
| ALERT2_SKIP | 96 |
| ALERT3 | 293 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 110 |
| PARTIAL | 18 |
| TARGET_HIT | 1 |
| STOP_HIT | 110 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 129 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 47 / 82
- **Target hits / Stop hits / Partials:** 1 / 110 / 18
- **Avg / median % per leg:** 0.67% / -0.75%
- **Sum % (uncompounded):** 86.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 55 | 19 | 34.5% | 1 | 49 | 5 | 0.36% | 19.7% |
| BUY @ 2nd Alert (retest1) | 10 | 10 | 100.0% | 0 | 5 | 5 | 4.29% | 42.9% |
| BUY @ 3rd Alert (retest2) | 45 | 9 | 20.0% | 1 | 44 | 0 | -0.52% | -23.3% |
| SELL (all) | 74 | 28 | 37.8% | 0 | 61 | 13 | 0.90% | 67.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 74 | 28 | 37.8% | 0 | 61 | 13 | 0.90% | 67.0% |
| retest1 (combined) | 10 | 10 | 100.0% | 0 | 5 | 5 | 4.29% | 42.9% |
| retest2 (combined) | 119 | 37 | 31.1% | 1 | 105 | 13 | 0.37% | 43.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 11:15:00 | 1467.10 | 1452.98 | 1451.87 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 10:15:00 | 1419.90 | 1447.27 | 1450.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-17 10:15:00 | 1374.35 | 1421.06 | 1435.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-18 11:15:00 | 1386.00 | 1380.63 | 1400.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 12:15:00 | 1364.45 | 1357.17 | 1364.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 12:15:00 | 1364.45 | 1357.17 | 1364.23 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-05-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 11:15:00 | 1364.10 | 1361.97 | 1361.78 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2023-05-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 13:15:00 | 1361.40 | 1361.61 | 1361.64 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 14:15:00 | 1363.00 | 1361.89 | 1361.76 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 15:15:00 | 1357.00 | 1360.91 | 1361.33 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1377.00 | 1364.13 | 1362.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 1386.40 | 1370.71 | 1366.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 1395.60 | 1395.94 | 1387.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 09:15:00 | 1405.70 | 1413.38 | 1411.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 09:15:00 | 1405.70 | 1413.38 | 1411.00 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2023-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 11:15:00 | 1400.00 | 1408.64 | 1409.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 14:15:00 | 1399.00 | 1404.34 | 1406.83 | Break + close below crossover candle low |

### Cycle 9 — BUY (started 2023-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 09:15:00 | 1433.20 | 1409.50 | 1408.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 10:15:00 | 1437.60 | 1415.12 | 1411.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 14:15:00 | 1444.00 | 1446.28 | 1438.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 1440.05 | 1444.26 | 1438.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 1440.05 | 1444.26 | 1438.68 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 1429.00 | 1436.21 | 1436.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 09:15:00 | 1425.00 | 1432.98 | 1434.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 1440.00 | 1425.31 | 1428.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 1440.00 | 1425.31 | 1428.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 1440.00 | 1425.31 | 1428.92 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 13:15:00 | 1439.95 | 1432.17 | 1431.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 14:15:00 | 1440.25 | 1433.79 | 1432.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 13:15:00 | 1444.80 | 1445.20 | 1439.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 12:15:00 | 1475.60 | 1485.80 | 1476.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 12:15:00 | 1475.60 | 1485.80 | 1476.09 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 15:15:00 | 1448.00 | 1470.08 | 1470.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 09:15:00 | 1445.05 | 1465.07 | 1468.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 1389.00 | 1381.69 | 1390.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 13:15:00 | 1385.75 | 1382.51 | 1389.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 1385.75 | 1382.51 | 1389.94 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-30 10:15:00 | 1384.00 | 1382.15 | 1381.99 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-06-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 11:15:00 | 1380.00 | 1381.72 | 1381.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 12:15:00 | 1375.45 | 1380.47 | 1381.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-03 09:15:00 | 1386.05 | 1379.07 | 1380.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-03 09:15:00 | 1386.05 | 1379.07 | 1380.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 1386.05 | 1379.07 | 1380.06 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 11:15:00 | 1383.70 | 1380.95 | 1380.80 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 12:15:00 | 1379.00 | 1380.56 | 1380.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 10:15:00 | 1374.00 | 1378.58 | 1379.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 09:15:00 | 1379.60 | 1376.24 | 1377.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 09:15:00 | 1379.60 | 1376.24 | 1377.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 1379.60 | 1376.24 | 1377.60 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-07 09:15:00 | 1424.60 | 1384.25 | 1379.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-07 11:15:00 | 1432.00 | 1399.55 | 1387.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 10:15:00 | 1418.30 | 1422.07 | 1406.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 11:15:00 | 1401.85 | 1418.03 | 1406.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 1401.85 | 1418.03 | 1406.19 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 1401.40 | 1419.99 | 1421.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 1392.00 | 1409.30 | 1415.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 1417.00 | 1403.70 | 1409.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 14:15:00 | 1417.00 | 1403.70 | 1409.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 1417.00 | 1403.70 | 1409.44 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 1426.50 | 1414.47 | 1413.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 1441.10 | 1419.79 | 1415.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 1414.00 | 1423.68 | 1419.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 1414.00 | 1423.68 | 1419.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 1414.00 | 1423.68 | 1419.89 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 1407.50 | 1416.05 | 1417.00 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 13:15:00 | 1427.45 | 1416.14 | 1415.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 14:15:00 | 1436.40 | 1420.19 | 1417.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-24 10:15:00 | 1480.90 | 1486.42 | 1467.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 11:15:00 | 1476.00 | 1484.33 | 1468.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 1476.00 | 1484.33 | 1468.09 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2023-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 11:15:00 | 1595.70 | 1602.54 | 1603.13 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 09:15:00 | 1641.00 | 1608.44 | 1605.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 09:15:00 | 1694.00 | 1649.53 | 1638.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 10:15:00 | 1706.25 | 1707.74 | 1693.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 1707.95 | 1717.02 | 1710.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1707.95 | 1717.02 | 1710.83 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 09:15:00 | 1697.60 | 1708.80 | 1709.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 09:15:00 | 1692.90 | 1700.64 | 1704.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 10:15:00 | 1707.00 | 1701.91 | 1704.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 10:15:00 | 1707.00 | 1701.91 | 1704.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 10:15:00 | 1707.00 | 1701.91 | 1704.55 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 15:15:00 | 1710.00 | 1706.25 | 1706.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-24 09:15:00 | 1717.75 | 1708.55 | 1707.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-25 14:15:00 | 1726.00 | 1734.34 | 1726.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 14:15:00 | 1726.00 | 1734.34 | 1726.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 14:15:00 | 1726.00 | 1734.34 | 1726.57 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2023-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 13:15:00 | 1813.50 | 1829.45 | 1831.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 14:15:00 | 1799.05 | 1823.37 | 1828.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 12:15:00 | 1783.65 | 1779.24 | 1792.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 1780.00 | 1778.50 | 1788.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1780.00 | 1778.50 | 1788.06 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 1724.50 | 1708.18 | 1707.34 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 1702.75 | 1713.33 | 1713.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 1692.05 | 1704.25 | 1709.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1703.45 | 1700.31 | 1705.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1703.45 | 1700.31 | 1705.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1703.45 | 1700.31 | 1705.19 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 1715.10 | 1705.48 | 1705.41 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 1699.95 | 1704.60 | 1705.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 09:15:00 | 1681.00 | 1700.00 | 1702.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-10 10:15:00 | 1677.70 | 1677.55 | 1686.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 11:15:00 | 1685.70 | 1679.18 | 1686.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 11:15:00 | 1685.70 | 1679.18 | 1686.66 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 10:15:00 | 1708.60 | 1692.94 | 1690.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 1730.05 | 1702.54 | 1696.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 11:15:00 | 1748.00 | 1750.81 | 1731.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-16 14:15:00 | 1741.05 | 1753.50 | 1746.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 1741.05 | 1753.50 | 1746.63 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 1738.80 | 1756.84 | 1758.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 09:15:00 | 1721.15 | 1749.71 | 1754.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1607.70 | 1591.72 | 1615.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 14:15:00 | 1604.95 | 1598.85 | 1610.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 14:15:00 | 1604.95 | 1598.85 | 1610.23 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 11:15:00 | 1610.05 | 1600.52 | 1600.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 1619.30 | 1609.68 | 1605.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 14:15:00 | 1611.60 | 1613.41 | 1609.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 15:15:00 | 1612.00 | 1613.13 | 1609.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 15:15:00 | 1612.00 | 1613.13 | 1609.69 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 09:15:00 | 1638.70 | 1659.63 | 1660.71 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-11-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 11:15:00 | 1656.00 | 1651.47 | 1651.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 12:15:00 | 1662.70 | 1653.72 | 1652.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 11:15:00 | 1659.35 | 1659.75 | 1656.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 11:15:00 | 1659.35 | 1659.75 | 1656.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 1659.35 | 1659.75 | 1656.37 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 13:15:00 | 1652.00 | 1669.50 | 1669.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 11:15:00 | 1647.55 | 1657.17 | 1662.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1653.95 | 1652.90 | 1658.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1653.95 | 1652.90 | 1658.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1653.95 | 1652.90 | 1658.13 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 11:15:00 | 1664.00 | 1658.82 | 1658.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 15:15:00 | 1670.05 | 1662.96 | 1660.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 1721.50 | 1743.61 | 1733.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 1721.50 | 1743.61 | 1733.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1721.50 | 1743.61 | 1733.65 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2023-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 13:15:00 | 1701.50 | 1724.22 | 1726.77 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 1729.00 | 1727.40 | 1727.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 14:15:00 | 1734.10 | 1729.45 | 1728.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 09:15:00 | 1724.95 | 1728.66 | 1728.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 1724.95 | 1728.66 | 1728.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1724.95 | 1728.66 | 1728.16 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 1744.20 | 1753.00 | 1753.05 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 11:15:00 | 1756.00 | 1753.14 | 1753.08 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2023-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 15:15:00 | 1746.05 | 1751.86 | 1752.56 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2023-12-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 09:15:00 | 1758.50 | 1753.19 | 1753.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 1770.10 | 1760.76 | 1757.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 1798.80 | 1801.55 | 1787.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-20 12:15:00 | 1798.60 | 1809.47 | 1801.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 12:15:00 | 1798.60 | 1809.47 | 1801.40 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1745.00 | 1788.24 | 1792.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 13:15:00 | 1717.70 | 1759.17 | 1774.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-27 14:15:00 | 1675.00 | 1666.23 | 1685.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 13:15:00 | 1660.00 | 1656.64 | 1665.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 1660.00 | 1656.64 | 1665.43 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2024-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 13:15:00 | 1669.55 | 1663.97 | 1663.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 09:15:00 | 1683.25 | 1669.12 | 1666.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 09:15:00 | 1665.65 | 1692.61 | 1683.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-04 09:15:00 | 1665.65 | 1692.61 | 1683.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 1665.65 | 1692.61 | 1683.06 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-01-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 14:15:00 | 1664.05 | 1676.35 | 1677.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-05 10:15:00 | 1656.50 | 1669.38 | 1673.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-10 15:15:00 | 1563.00 | 1554.93 | 1575.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-12 09:15:00 | 1557.00 | 1541.63 | 1556.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 09:15:00 | 1557.00 | 1541.63 | 1556.12 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 11:15:00 | 1535.20 | 1507.37 | 1506.55 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 11:15:00 | 1492.85 | 1512.01 | 1513.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 13:15:00 | 1486.05 | 1503.45 | 1509.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 13:15:00 | 1458.20 | 1454.43 | 1468.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 1474.35 | 1459.91 | 1467.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1474.35 | 1459.91 | 1467.48 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 12:15:00 | 1491.40 | 1474.89 | 1473.20 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 15:15:00 | 1464.70 | 1476.41 | 1477.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-31 11:15:00 | 1458.40 | 1468.56 | 1473.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 12:15:00 | 1468.75 | 1468.60 | 1472.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 14:15:00 | 1454.85 | 1465.83 | 1470.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 1454.85 | 1465.83 | 1470.83 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-08 13:15:00 | 1426.00 | 1419.27 | 1418.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-08 14:15:00 | 1430.05 | 1421.43 | 1419.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 09:15:00 | 1408.65 | 1419.61 | 1419.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 1408.65 | 1419.61 | 1419.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1408.65 | 1419.61 | 1419.38 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 1390.45 | 1413.78 | 1416.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 11:15:00 | 1382.00 | 1397.37 | 1405.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 15:15:00 | 1368.50 | 1367.05 | 1379.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 09:15:00 | 1370.95 | 1369.51 | 1374.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 09:15:00 | 1370.95 | 1369.51 | 1374.60 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 1378.30 | 1373.17 | 1372.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 1384.50 | 1379.10 | 1375.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 14:15:00 | 1380.20 | 1382.85 | 1379.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 1382.00 | 1383.02 | 1379.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1382.00 | 1383.02 | 1379.98 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 14:15:00 | 1375.10 | 1378.24 | 1378.52 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 1394.35 | 1381.26 | 1379.83 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 1376.25 | 1382.43 | 1382.50 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 1394.35 | 1383.66 | 1382.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 1397.25 | 1387.76 | 1384.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 15:15:00 | 1385.15 | 1389.14 | 1386.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 15:15:00 | 1385.15 | 1389.14 | 1386.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 15:15:00 | 1385.15 | 1389.14 | 1386.23 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1376.60 | 1394.42 | 1395.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1368.00 | 1386.03 | 1391.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 1372.30 | 1368.27 | 1376.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 15:15:00 | 1370.00 | 1368.62 | 1375.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1370.00 | 1368.62 | 1375.93 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-03-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 12:15:00 | 1392.00 | 1379.56 | 1379.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 1402.90 | 1389.33 | 1384.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 11:15:00 | 1390.15 | 1392.99 | 1388.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 11:15:00 | 1390.15 | 1392.99 | 1388.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 11:15:00 | 1390.15 | 1392.99 | 1388.47 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 1383.65 | 1386.35 | 1386.51 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 13:15:00 | 1388.90 | 1386.75 | 1386.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 14:15:00 | 1389.90 | 1387.38 | 1386.94 | Break + close above crossover candle high |

### Cycle 62 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 1368.35 | 1384.26 | 1385.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 1359.60 | 1376.37 | 1381.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 1378.70 | 1375.96 | 1380.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 1378.70 | 1375.96 | 1380.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1378.70 | 1375.96 | 1380.04 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 1396.50 | 1382.59 | 1382.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 12:15:00 | 1403.60 | 1389.03 | 1385.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 1394.35 | 1401.18 | 1393.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 1394.35 | 1401.18 | 1393.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1394.35 | 1401.18 | 1393.37 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 1378.35 | 1389.79 | 1390.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 1368.00 | 1385.43 | 1388.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 10:15:00 | 1321.15 | 1318.23 | 1337.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 1326.85 | 1320.35 | 1330.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1326.85 | 1320.35 | 1330.27 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 1310.80 | 1294.00 | 1292.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 1314.65 | 1298.13 | 1294.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 1321.40 | 1323.74 | 1314.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 1321.40 | 1323.74 | 1314.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1321.40 | 1323.74 | 1314.30 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2024-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 13:15:00 | 1373.05 | 1383.29 | 1383.73 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 1395.30 | 1386.03 | 1384.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 12:15:00 | 1408.00 | 1392.78 | 1388.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 1389.05 | 1405.65 | 1401.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 1389.05 | 1405.65 | 1401.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1389.05 | 1405.65 | 1401.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 1381.60 | 1405.65 | 1401.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 11:15:00 | 1388.15 | 1401.80 | 1400.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:45:00 | 1388.00 | 1401.80 | 1400.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 12:15:00 | 1396.30 | 1400.70 | 1400.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:15:00 | 1397.50 | 1400.70 | 1400.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 1396.50 | 1399.16 | 1399.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 1396.50 | 1399.16 | 1399.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 15:15:00 | 1390.00 | 1397.33 | 1398.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1401.70 | 1398.20 | 1398.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1401.70 | 1398.20 | 1398.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1401.70 | 1398.20 | 1398.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 1401.70 | 1398.20 | 1398.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 10:15:00 | 1407.90 | 1400.14 | 1399.66 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 12:15:00 | 1384.80 | 1396.60 | 1398.11 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 1432.00 | 1402.63 | 1400.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 09:15:00 | 1435.10 | 1426.25 | 1419.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 15:15:00 | 1433.30 | 1433.86 | 1426.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 09:15:00 | 1425.00 | 1433.86 | 1426.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1419.50 | 1430.99 | 1426.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:45:00 | 1418.00 | 1430.99 | 1426.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1414.55 | 1427.70 | 1425.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:45:00 | 1415.70 | 1427.70 | 1425.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 13:15:00 | 1419.45 | 1423.97 | 1423.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 13:45:00 | 1416.85 | 1423.97 | 1423.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2024-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 14:15:00 | 1421.90 | 1423.56 | 1423.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 12:15:00 | 1409.50 | 1419.23 | 1421.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 12:15:00 | 1411.90 | 1411.03 | 1415.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-25 13:00:00 | 1411.90 | 1411.03 | 1415.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 14:15:00 | 1410.40 | 1410.90 | 1414.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 15:15:00 | 1411.90 | 1410.90 | 1414.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 15:15:00 | 1411.90 | 1411.10 | 1414.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 09:15:00 | 1416.65 | 1411.10 | 1414.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1404.85 | 1409.85 | 1413.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 10:15:00 | 1402.55 | 1409.85 | 1413.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 13:30:00 | 1403.10 | 1408.47 | 1411.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-26 14:00:00 | 1402.80 | 1408.47 | 1411.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:15:00 | 1332.42 | 1345.68 | 1360.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:15:00 | 1332.94 | 1345.68 | 1360.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-03 09:15:00 | 1332.66 | 1345.68 | 1360.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-06 13:15:00 | 1313.00 | 1312.79 | 1328.85 | SL hit (close>ema200) qty=0.50 sl=1312.79 alert=retest2 |

### Cycle 73 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 1315.25 | 1303.14 | 1302.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1320.00 | 1308.81 | 1304.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 13:15:00 | 1292.25 | 1312.16 | 1308.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 13:15:00 | 1292.25 | 1312.16 | 1308.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 1292.25 | 1312.16 | 1308.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:30:00 | 1326.00 | 1312.16 | 1308.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1298.40 | 1309.41 | 1307.57 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 1278.25 | 1301.19 | 1304.03 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 11:15:00 | 1318.90 | 1301.05 | 1299.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 14:15:00 | 1321.00 | 1307.43 | 1303.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 09:15:00 | 1340.20 | 1341.96 | 1332.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 09:15:00 | 1340.20 | 1341.96 | 1332.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1340.20 | 1341.96 | 1332.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1339.40 | 1341.96 | 1332.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1339.00 | 1342.82 | 1338.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:45:00 | 1335.90 | 1342.82 | 1338.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1341.75 | 1342.61 | 1338.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 1330.70 | 1342.61 | 1338.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1341.15 | 1342.31 | 1338.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 14:00:00 | 1350.75 | 1344.16 | 1340.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 15:00:00 | 1349.10 | 1345.15 | 1340.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1334.05 | 1340.73 | 1341.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1334.05 | 1340.73 | 1341.20 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 1346.10 | 1341.80 | 1341.64 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 11:15:00 | 1336.10 | 1340.66 | 1341.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 1320.10 | 1336.02 | 1338.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 13:15:00 | 1339.25 | 1335.97 | 1338.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 13:15:00 | 1339.25 | 1335.97 | 1338.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1339.25 | 1335.97 | 1338.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 1339.25 | 1335.97 | 1338.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 1334.95 | 1335.76 | 1337.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 1322.80 | 1336.01 | 1336.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:00:00 | 1328.40 | 1316.96 | 1321.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 1334.30 | 1323.99 | 1323.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1334.30 | 1323.99 | 1323.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 1334.90 | 1326.17 | 1324.65 | Break + close above crossover candle high |

### Cycle 80 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 1309.70 | 1324.24 | 1324.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1275.00 | 1314.39 | 1319.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 1311.45 | 1290.69 | 1302.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 1311.45 | 1290.69 | 1302.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1311.45 | 1290.69 | 1302.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 1311.45 | 1290.69 | 1302.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1320.00 | 1296.55 | 1303.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 1320.00 | 1296.55 | 1303.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 13:15:00 | 1325.25 | 1310.85 | 1309.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1331.35 | 1317.83 | 1313.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 1338.45 | 1338.46 | 1330.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 1338.45 | 1338.46 | 1330.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1345.75 | 1340.01 | 1333.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:15:00 | 1350.90 | 1340.01 | 1333.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 14:45:00 | 1348.05 | 1344.15 | 1338.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1361.20 | 1344.47 | 1338.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 12:15:00 | 1384.75 | 1390.27 | 1391.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 1384.75 | 1390.27 | 1391.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 13:15:00 | 1380.20 | 1388.26 | 1390.04 | Break + close below crossover candle low |

### Cycle 83 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1423.00 | 1394.14 | 1392.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 10:15:00 | 1439.90 | 1403.29 | 1396.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 1425.25 | 1425.97 | 1413.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 09:30:00 | 1420.85 | 1425.97 | 1413.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1439.70 | 1428.06 | 1420.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 10:30:00 | 1459.10 | 1431.35 | 1422.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:30:00 | 1444.70 | 1435.18 | 1428.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 1426.45 | 1445.28 | 1447.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 1426.45 | 1445.28 | 1447.05 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1485.40 | 1451.02 | 1449.21 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 1463.05 | 1467.97 | 1468.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 15:15:00 | 1459.00 | 1464.42 | 1466.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 1472.00 | 1465.94 | 1467.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 1472.00 | 1465.94 | 1467.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1472.00 | 1465.94 | 1467.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 10:00:00 | 1472.00 | 1465.94 | 1467.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 1469.45 | 1466.64 | 1467.27 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 11:15:00 | 1481.85 | 1469.68 | 1468.59 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 1462.20 | 1469.68 | 1470.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 1459.80 | 1467.70 | 1469.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 1462.90 | 1461.20 | 1464.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 10:15:00 | 1462.90 | 1461.20 | 1464.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1462.90 | 1461.20 | 1464.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:45:00 | 1464.75 | 1461.20 | 1464.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 1459.10 | 1460.78 | 1463.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 13:00:00 | 1458.10 | 1460.25 | 1463.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:15:00 | 1458.60 | 1460.10 | 1462.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 14:45:00 | 1454.55 | 1460.10 | 1462.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 15:15:00 | 1456.35 | 1460.10 | 1462.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1442.00 | 1455.88 | 1460.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 1429.05 | 1455.88 | 1460.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 12:15:00 | 1438.45 | 1449.34 | 1456.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 11:15:00 | 1470.90 | 1453.48 | 1454.61 | SL hit (close>static) qty=1.00 sl=1464.60 alert=retest2 |

### Cycle 89 — BUY (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 13:15:00 | 1460.05 | 1455.99 | 1455.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 1491.00 | 1464.32 | 1459.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 15:15:00 | 1473.05 | 1475.52 | 1468.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-15 09:15:00 | 1480.00 | 1475.52 | 1468.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1493.90 | 1479.20 | 1471.00 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 09:15:00 | 1461.05 | 1468.84 | 1469.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 11:15:00 | 1445.00 | 1461.90 | 1466.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1431.95 | 1413.00 | 1422.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1431.95 | 1413.00 | 1422.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1431.95 | 1413.00 | 1422.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 1419.00 | 1413.00 | 1422.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1422.05 | 1414.81 | 1422.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1432.20 | 1414.81 | 1422.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1432.65 | 1418.37 | 1423.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 1438.45 | 1418.37 | 1423.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1432.25 | 1421.15 | 1424.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 14:00:00 | 1425.80 | 1422.08 | 1424.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 15:15:00 | 1426.00 | 1424.26 | 1425.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 1469.45 | 1428.61 | 1424.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1469.45 | 1428.61 | 1424.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 1480.00 | 1445.32 | 1432.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 15:15:00 | 1488.05 | 1490.22 | 1472.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 09:15:00 | 1484.95 | 1490.22 | 1472.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 1512.65 | 1517.56 | 1512.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:30:00 | 1508.95 | 1517.56 | 1512.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 1503.50 | 1514.75 | 1511.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 1503.50 | 1514.75 | 1511.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1501.00 | 1512.00 | 1510.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 1501.00 | 1512.00 | 1510.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-07-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 15:15:00 | 1502.40 | 1508.15 | 1508.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 09:15:00 | 1490.65 | 1504.65 | 1507.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1432.10 | 1431.39 | 1453.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 1432.10 | 1431.39 | 1453.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1455.30 | 1427.46 | 1438.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 1455.30 | 1427.46 | 1438.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1453.50 | 1432.67 | 1439.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:30:00 | 1456.45 | 1432.67 | 1439.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1460.50 | 1445.36 | 1444.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 1462.00 | 1448.69 | 1445.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1453.55 | 1454.02 | 1449.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:00:00 | 1453.55 | 1454.02 | 1449.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1450.50 | 1453.32 | 1449.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 1450.50 | 1453.32 | 1449.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1454.95 | 1453.65 | 1450.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:30:00 | 1445.95 | 1453.65 | 1450.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1458.30 | 1454.58 | 1450.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1474.55 | 1454.58 | 1450.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 13:15:00 | 1462.70 | 1476.63 | 1476.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 13:15:00 | 1462.70 | 1476.63 | 1476.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1451.45 | 1471.60 | 1474.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 15:15:00 | 1443.35 | 1442.73 | 1454.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:15:00 | 1487.20 | 1442.73 | 1454.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1488.05 | 1451.80 | 1457.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 1484.80 | 1451.80 | 1457.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1491.30 | 1459.70 | 1461.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 1491.30 | 1459.70 | 1461.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1497.40 | 1467.24 | 1464.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1499.40 | 1482.35 | 1472.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 1505.80 | 1512.56 | 1499.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 09:45:00 | 1505.25 | 1512.56 | 1499.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1496.00 | 1509.25 | 1498.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 1494.80 | 1509.25 | 1498.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1505.95 | 1508.59 | 1499.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 14:00:00 | 1511.50 | 1508.37 | 1500.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1516.90 | 1507.46 | 1501.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 1497.70 | 1511.17 | 1512.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 1497.70 | 1511.17 | 1512.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 1490.90 | 1504.95 | 1508.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1500.75 | 1497.00 | 1503.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 1500.75 | 1497.00 | 1503.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1500.75 | 1497.00 | 1503.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:00:00 | 1500.75 | 1497.00 | 1503.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1499.30 | 1497.46 | 1502.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 10:45:00 | 1500.05 | 1497.46 | 1502.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1500.65 | 1498.10 | 1502.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:30:00 | 1498.65 | 1498.10 | 1502.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1511.35 | 1500.75 | 1503.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 1511.35 | 1500.75 | 1503.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 1511.90 | 1502.98 | 1504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 1514.00 | 1502.98 | 1504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 1515.00 | 1505.38 | 1505.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 10:15:00 | 1519.00 | 1509.96 | 1507.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 1521.30 | 1527.45 | 1521.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 14:15:00 | 1521.30 | 1527.45 | 1521.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1521.30 | 1527.45 | 1521.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1521.30 | 1527.45 | 1521.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1521.00 | 1526.16 | 1521.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1517.85 | 1526.16 | 1521.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1522.65 | 1525.46 | 1521.39 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 1506.15 | 1516.65 | 1517.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 1501.00 | 1513.52 | 1516.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1526.35 | 1514.70 | 1516.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1526.35 | 1514.70 | 1516.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1526.35 | 1514.70 | 1516.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1526.35 | 1514.70 | 1516.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1525.50 | 1516.86 | 1516.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:15:00 | 1526.95 | 1516.86 | 1516.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 1520.05 | 1517.50 | 1517.25 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 1513.10 | 1517.03 | 1517.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 1509.80 | 1513.81 | 1515.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 1515.55 | 1514.16 | 1515.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 1515.55 | 1514.16 | 1515.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1515.55 | 1514.16 | 1515.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1515.55 | 1514.16 | 1515.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1512.10 | 1513.75 | 1514.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1513.75 | 1513.75 | 1514.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 09:15:00 | 1535.15 | 1518.03 | 1516.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1560.45 | 1530.54 | 1523.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 09:15:00 | 1559.50 | 1563.40 | 1547.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 10:00:00 | 1559.50 | 1563.40 | 1547.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1563.45 | 1561.44 | 1552.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:30:00 | 1557.60 | 1561.44 | 1552.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1551.00 | 1561.20 | 1553.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1547.15 | 1561.20 | 1553.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1557.15 | 1560.39 | 1554.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 14:45:00 | 1563.05 | 1559.19 | 1555.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 1647.80 | 1664.80 | 1666.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 1647.80 | 1664.80 | 1666.28 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 1688.50 | 1668.36 | 1665.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 1697.95 | 1674.28 | 1668.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 1701.55 | 1701.79 | 1692.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 11:45:00 | 1700.20 | 1701.79 | 1692.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1693.80 | 1704.33 | 1697.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 1693.80 | 1704.33 | 1697.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1697.30 | 1702.93 | 1697.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:15:00 | 1701.85 | 1702.93 | 1697.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:45:00 | 1703.80 | 1709.22 | 1707.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 14:15:00 | 1686.30 | 1704.64 | 1705.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1686.30 | 1704.64 | 1705.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 15:15:00 | 1679.00 | 1699.51 | 1702.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1631.10 | 1630.21 | 1642.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 1631.10 | 1630.21 | 1642.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1584.35 | 1572.72 | 1595.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 1592.95 | 1572.72 | 1595.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 1569.00 | 1575.07 | 1592.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 12:15:00 | 1566.10 | 1575.07 | 1592.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 1605.60 | 1583.59 | 1593.77 | SL hit (close>static) qty=1.00 sl=1596.90 alert=retest2 |

### Cycle 105 — BUY (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 12:15:00 | 1604.50 | 1599.12 | 1598.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1611.20 | 1601.77 | 1599.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 11:15:00 | 1597.05 | 1601.28 | 1600.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 11:15:00 | 1597.05 | 1601.28 | 1600.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1597.05 | 1601.28 | 1600.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 1597.05 | 1601.28 | 1600.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 1600.10 | 1601.04 | 1600.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:15:00 | 1602.45 | 1601.04 | 1600.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 13:45:00 | 1604.50 | 1600.76 | 1600.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 14:45:00 | 1605.05 | 1601.75 | 1600.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 10:15:00 | 1592.50 | 1608.03 | 1607.28 | SL hit (close<static) qty=1.00 sl=1593.20 alert=retest2 |

### Cycle 106 — SELL (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 11:15:00 | 1592.80 | 1604.98 | 1605.97 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1644.25 | 1608.11 | 1605.56 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1578.15 | 1604.03 | 1605.57 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 1626.20 | 1609.24 | 1607.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 10:15:00 | 1638.00 | 1618.23 | 1613.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-18 12:15:00 | 1616.85 | 1623.12 | 1617.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 12:15:00 | 1616.85 | 1623.12 | 1617.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1616.85 | 1623.12 | 1617.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:00:00 | 1616.85 | 1623.12 | 1617.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1616.00 | 1621.70 | 1617.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:00:00 | 1616.00 | 1621.70 | 1617.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1609.40 | 1619.24 | 1616.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:00:00 | 1609.40 | 1619.24 | 1616.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1606.15 | 1616.62 | 1615.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1589.20 | 1616.62 | 1615.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 1591.90 | 1611.68 | 1613.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 13:15:00 | 1575.05 | 1593.09 | 1603.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 1498.95 | 1493.62 | 1510.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:15:00 | 1505.80 | 1493.62 | 1510.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 1512.35 | 1497.36 | 1510.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:15:00 | 1517.15 | 1497.36 | 1510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 1531.10 | 1504.11 | 1512.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:45:00 | 1531.50 | 1504.11 | 1512.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 1529.90 | 1509.27 | 1513.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 1529.85 | 1509.27 | 1513.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 1537.00 | 1519.91 | 1518.24 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1509.45 | 1516.66 | 1517.56 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 15:15:00 | 1520.00 | 1518.06 | 1517.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 1541.60 | 1522.76 | 1520.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 1553.00 | 1554.11 | 1542.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 12:45:00 | 1551.50 | 1554.11 | 1542.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1483.85 | 1548.99 | 1546.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1483.85 | 1548.99 | 1546.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1459.05 | 1531.00 | 1538.61 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 1519.00 | 1512.94 | 1512.27 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1507.05 | 1511.37 | 1511.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 1489.25 | 1504.93 | 1508.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1482.50 | 1479.18 | 1489.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 1482.50 | 1479.18 | 1489.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1494.05 | 1475.55 | 1482.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 1495.20 | 1475.55 | 1482.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1494.80 | 1479.40 | 1483.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 1488.20 | 1479.40 | 1483.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 11:15:00 | 1487.30 | 1465.30 | 1462.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1487.30 | 1465.30 | 1462.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 1492.70 | 1470.78 | 1465.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1474.15 | 1474.43 | 1468.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 1474.15 | 1474.43 | 1468.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1448.35 | 1469.29 | 1466.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 1448.35 | 1469.29 | 1466.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1452.10 | 1465.85 | 1465.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 1448.35 | 1465.85 | 1465.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 1454.15 | 1463.51 | 1464.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 12:15:00 | 1448.00 | 1460.41 | 1462.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 1452.60 | 1452.09 | 1456.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 12:00:00 | 1452.60 | 1452.09 | 1456.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1453.55 | 1452.09 | 1455.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1453.55 | 1452.09 | 1455.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1464.45 | 1454.56 | 1456.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 1468.45 | 1454.56 | 1456.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 1465.00 | 1456.65 | 1457.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 1488.00 | 1456.65 | 1457.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1487.30 | 1462.78 | 1460.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 13:15:00 | 1512.95 | 1483.68 | 1477.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1570.50 | 1591.02 | 1582.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 1570.50 | 1591.02 | 1582.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1570.50 | 1591.02 | 1582.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 1566.05 | 1591.02 | 1582.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1563.60 | 1585.54 | 1580.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1563.60 | 1585.54 | 1580.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1579.70 | 1581.26 | 1579.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:45:00 | 1571.60 | 1581.26 | 1579.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1580.00 | 1581.00 | 1579.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 1559.00 | 1581.00 | 1579.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1534.35 | 1571.67 | 1575.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 1501.00 | 1543.28 | 1557.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 13:15:00 | 1491.65 | 1491.29 | 1511.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 14:00:00 | 1491.65 | 1491.29 | 1511.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1482.15 | 1485.12 | 1497.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1482.15 | 1485.12 | 1497.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1477.60 | 1460.88 | 1467.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 1475.95 | 1460.88 | 1467.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 1472.80 | 1463.26 | 1468.29 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2024-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 15:15:00 | 1474.05 | 1471.34 | 1471.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1523.35 | 1481.74 | 1475.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 1489.60 | 1490.08 | 1482.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:30:00 | 1490.20 | 1490.08 | 1482.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1451.45 | 1482.68 | 1481.12 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 1436.85 | 1473.52 | 1477.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 1433.95 | 1465.60 | 1473.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 1421.65 | 1415.14 | 1433.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 1421.65 | 1415.14 | 1433.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1310.90 | 1307.63 | 1318.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 09:30:00 | 1315.45 | 1307.63 | 1318.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 1315.00 | 1310.01 | 1316.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:45:00 | 1315.30 | 1310.01 | 1316.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 1316.70 | 1311.35 | 1316.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:45:00 | 1315.65 | 1311.35 | 1316.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 1313.30 | 1311.74 | 1315.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 1306.75 | 1311.74 | 1315.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 1308.50 | 1311.09 | 1315.16 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 1320.00 | 1316.39 | 1316.28 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 09:15:00 | 1315.10 | 1316.13 | 1316.18 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-01-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 10:15:00 | 1321.10 | 1317.12 | 1316.62 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 1308.95 | 1315.67 | 1316.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 1304.50 | 1313.44 | 1315.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 1225.00 | 1222.70 | 1250.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 1225.00 | 1222.70 | 1250.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1086.30 | 1089.74 | 1108.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 13:30:00 | 1081.90 | 1090.01 | 1102.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 14:00:00 | 1082.35 | 1090.01 | 1102.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 14:45:00 | 1081.95 | 1088.53 | 1095.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 1077.50 | 1087.32 | 1090.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1080.50 | 1071.20 | 1077.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-23 14:15:00 | 1093.70 | 1082.17 | 1080.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 14:15:00 | 1093.70 | 1082.17 | 1080.81 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1071.80 | 1079.53 | 1080.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 1063.25 | 1074.11 | 1077.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 1044.50 | 1041.70 | 1053.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 1044.50 | 1041.70 | 1053.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 1055.65 | 1046.75 | 1052.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 1057.00 | 1046.75 | 1052.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 1055.45 | 1048.49 | 1052.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 1053.90 | 1048.49 | 1052.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1070.00 | 1052.79 | 1054.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1070.00 | 1052.79 | 1054.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 1070.70 | 1056.38 | 1055.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 1078.50 | 1065.50 | 1060.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 1072.30 | 1074.23 | 1068.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:00:00 | 1072.30 | 1074.23 | 1068.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1062.75 | 1071.94 | 1067.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1062.75 | 1071.94 | 1067.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1069.50 | 1071.45 | 1068.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 1064.25 | 1071.45 | 1068.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 1067.20 | 1070.60 | 1067.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 09:15:00 | 1072.50 | 1070.60 | 1067.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:00:00 | 1078.40 | 1083.39 | 1078.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 09:15:00 | 1077.85 | 1115.27 | 1116.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 1077.85 | 1115.27 | 1116.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1065.70 | 1078.80 | 1089.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 1070.95 | 1058.74 | 1070.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 1070.95 | 1058.74 | 1070.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1070.95 | 1058.74 | 1070.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1070.95 | 1058.74 | 1070.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1066.45 | 1060.28 | 1069.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 13:45:00 | 1046.65 | 1058.18 | 1067.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 10:15:00 | 1056.50 | 1058.29 | 1065.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 11:30:00 | 1060.45 | 1060.95 | 1065.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 1060.50 | 1062.69 | 1065.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1044.75 | 1058.67 | 1063.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1040.90 | 1058.67 | 1063.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 1007.43 | 1036.26 | 1050.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 13:15:00 | 1007.47 | 1036.26 | 1050.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1003.67 | 1024.45 | 1040.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 10:15:00 | 994.32 | 1006.36 | 1020.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 12:15:00 | 988.86 | 1002.12 | 1016.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 14:15:00 | 990.25 | 987.99 | 998.81 | SL hit (close>ema200) qty=0.50 sl=987.99 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1011.85 | 1002.41 | 1002.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 13:15:00 | 1015.70 | 1010.31 | 1006.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 1009.80 | 1010.67 | 1007.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 09:15:00 | 998.15 | 1010.67 | 1007.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1003.10 | 1009.15 | 1007.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 996.35 | 1009.15 | 1007.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 10:15:00 | 1011.90 | 1009.70 | 1007.67 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 999.10 | 1005.76 | 1006.44 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 09:15:00 | 1017.90 | 1007.25 | 1006.94 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 1001.65 | 1006.13 | 1006.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 979.70 | 1000.84 | 1004.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 922.50 | 903.62 | 916.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 922.50 | 903.62 | 916.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 922.50 | 903.62 | 916.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:45:00 | 923.20 | 903.62 | 916.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 923.10 | 907.52 | 916.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:45:00 | 925.60 | 907.52 | 916.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 924.95 | 911.00 | 917.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:00:00 | 924.95 | 911.00 | 917.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 928.30 | 918.85 | 920.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 15:00:00 | 928.30 | 918.85 | 920.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 15:15:00 | 929.00 | 920.88 | 920.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 946.60 | 926.03 | 923.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 970.85 | 974.64 | 962.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:00:00 | 970.85 | 974.64 | 962.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 962.75 | 972.26 | 962.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:00:00 | 962.75 | 972.26 | 962.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 963.10 | 970.43 | 962.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:15:00 | 962.00 | 970.43 | 962.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 962.00 | 968.74 | 962.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 964.30 | 968.74 | 962.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 964.85 | 967.96 | 962.61 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 940.70 | 959.51 | 960.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 914.45 | 947.38 | 954.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 939.05 | 937.61 | 945.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 939.05 | 937.61 | 945.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 939.05 | 937.61 | 945.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:45:00 | 932.00 | 935.84 | 943.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:15:00 | 885.40 | 897.53 | 910.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 900.65 | 897.57 | 908.07 | SL hit (close>ema200) qty=0.50 sl=897.57 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 934.95 | 916.08 | 914.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 943.25 | 921.51 | 916.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 966.90 | 969.38 | 955.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 12:00:00 | 966.90 | 969.38 | 955.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 957.30 | 965.45 | 956.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:30:00 | 955.50 | 965.45 | 956.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 14:15:00 | 959.95 | 964.35 | 956.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:45:00 | 963.00 | 963.98 | 957.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:45:00 | 964.15 | 973.39 | 970.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 11:15:00 | 950.00 | 966.16 | 967.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 950.00 | 966.16 | 967.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 933.00 | 950.82 | 958.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 13:15:00 | 940.30 | 939.98 | 950.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 14:00:00 | 940.30 | 939.98 | 950.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 939.40 | 934.30 | 943.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 941.15 | 934.30 | 943.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 934.00 | 930.11 | 937.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:30:00 | 918.90 | 926.58 | 933.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 955.25 | 929.57 | 930.91 | SL hit (close>static) qty=1.00 sl=950.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 11:15:00 | 953.65 | 934.38 | 932.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 12:15:00 | 962.10 | 939.93 | 935.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-02 09:15:00 | 947.50 | 949.07 | 942.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 09:15:00 | 947.50 | 949.07 | 942.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 947.50 | 949.07 | 942.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 947.70 | 949.07 | 942.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 957.50 | 961.47 | 956.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 936.35 | 961.47 | 956.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 919.70 | 953.12 | 953.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 10:15:00 | 916.45 | 945.78 | 950.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 864.30 | 863.16 | 886.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 14:15:00 | 884.75 | 868.94 | 881.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 884.75 | 868.94 | 881.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 15:00:00 | 884.75 | 868.94 | 881.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 15:15:00 | 888.80 | 872.92 | 882.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 873.10 | 872.92 | 882.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 916.80 | 879.24 | 879.51 | SL hit (close>static) qty=1.00 sl=888.80 alert=retest2 |

### Cycle 141 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 921.25 | 887.65 | 883.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 940.65 | 913.85 | 900.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 945.55 | 946.20 | 934.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 12:45:00 | 953.50 | 948.59 | 938.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 14:00:00 | 953.50 | 949.57 | 939.98 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 12:15:00 | 953.10 | 951.62 | 945.05 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 13:00:00 | 955.55 | 952.40 | 946.01 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 971.45 | 969.89 | 963.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 965.45 | 969.89 | 963.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:15:00 | 1001.18 | 983.83 | 974.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:15:00 | 1001.18 | 983.83 | 974.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:15:00 | 1000.76 | 983.83 | 974.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 12:15:00 | 1003.33 | 990.23 | 978.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 984.15 | 997.62 | 986.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 984.15 | 997.62 | 986.70 | SL hit (close<ema200) qty=0.50 sl=997.62 alert=retest1 |

### Cycle 142 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 972.40 | 981.34 | 981.85 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 985.75 | 982.86 | 982.49 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 15:15:00 | 981.00 | 982.63 | 982.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 09:15:00 | 977.70 | 981.65 | 982.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 951.00 | 949.93 | 957.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:30:00 | 951.45 | 949.93 | 957.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 953.65 | 950.67 | 957.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 15:00:00 | 953.65 | 950.67 | 957.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 955.00 | 951.54 | 957.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 09:15:00 | 947.55 | 951.54 | 957.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 946.05 | 952.42 | 954.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 949.05 | 940.39 | 940.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 949.05 | 940.39 | 940.37 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 935.15 | 940.09 | 940.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 930.30 | 938.13 | 939.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 925.00 | 924.73 | 930.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 945.00 | 924.73 | 930.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 960.70 | 931.93 | 932.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 960.70 | 931.93 | 932.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 963.60 | 938.26 | 935.72 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 937.00 | 943.77 | 944.40 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 949.95 | 945.01 | 944.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 959.65 | 947.94 | 946.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:30:00 | 986.60 | 959.83 | 953.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 969.55 | 992.22 | 990.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 969.85 | 987.75 | 988.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 969.85 | 987.75 | 988.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 958.05 | 981.81 | 985.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 15:15:00 | 972.40 | 972.34 | 979.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:15:00 | 958.50 | 972.34 | 979.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 955.00 | 960.42 | 968.40 | EMA400 retest candle locked (from downside) |

### Cycle 151 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 977.95 | 967.36 | 967.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 983.00 | 976.20 | 972.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1008.85 | 993.46 | 986.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 1003.85 | 1004.25 | 999.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:45:00 | 1001.60 | 1002.91 | 1000.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 1001.55 | 1001.57 | 1000.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 993.45 | 999.95 | 999.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 993.45 | 999.95 | 999.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1037.20 | 1005.20 | 1001.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 14:15:00 | 1042.20 | 1015.23 | 1009.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 1051.35 | 1052.14 | 1038.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:45:00 | 1053.65 | 1052.14 | 1038.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1037.40 | 1046.36 | 1039.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1039.60 | 1046.36 | 1039.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1039.85 | 1045.06 | 1039.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 1036.60 | 1045.06 | 1039.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1040.15 | 1044.08 | 1039.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1031.00 | 1044.08 | 1039.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1027.95 | 1040.85 | 1038.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 1028.75 | 1040.85 | 1038.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1027.70 | 1038.22 | 1037.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 1027.70 | 1038.22 | 1037.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 1026.50 | 1035.88 | 1036.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 1020.25 | 1029.97 | 1033.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 1008.00 | 1011.53 | 1016.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 957.60 | 983.62 | 995.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 958.25 | 957.55 | 969.22 | SL hit (close>ema200) qty=0.50 sl=957.55 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 14:15:00 | 962.80 | 959.30 | 959.20 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 952.00 | 957.84 | 958.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 939.20 | 951.58 | 955.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 11:15:00 | 947.20 | 947.06 | 951.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 11:45:00 | 947.25 | 947.06 | 951.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 948.90 | 947.31 | 950.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 948.90 | 947.31 | 950.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 952.70 | 948.39 | 950.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 949.70 | 948.39 | 950.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 951.95 | 949.10 | 951.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 955.70 | 949.10 | 951.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 956.40 | 950.56 | 951.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 963.90 | 950.56 | 951.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 972.00 | 954.85 | 953.37 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 959.00 | 965.05 | 965.59 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 969.00 | 966.23 | 965.98 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 957.00 | 964.38 | 965.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 956.50 | 960.37 | 962.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 960.90 | 959.81 | 961.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 959.00 | 959.65 | 961.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 957.00 | 959.65 | 961.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 961.30 | 959.98 | 961.19 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 969.65 | 963.28 | 962.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 974.95 | 966.57 | 964.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 971.80 | 972.63 | 968.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 971.60 | 972.63 | 968.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 964.55 | 972.31 | 970.16 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 964.30 | 968.60 | 968.81 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 980.55 | 970.99 | 969.88 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 965.55 | 969.00 | 969.43 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 993.80 | 973.87 | 971.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1002.95 | 979.69 | 974.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:15:00 | 990.05 | 1005.59 | 997.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 988.95 | 1002.26 | 996.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 988.95 | 1002.26 | 996.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 988.35 | 999.48 | 996.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:15:00 | 988.85 | 999.48 | 996.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 986.00 | 993.46 | 993.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 971.90 | 989.15 | 991.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 997.30 | 984.02 | 986.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 990.60 | 985.34 | 986.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 987.80 | 985.34 | 986.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1005.60 | 989.39 | 988.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1005.60 | 989.39 | 988.55 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 978.55 | 987.23 | 988.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 975.85 | 978.08 | 981.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 987.75 | 974.08 | 977.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 1008.65 | 980.99 | 980.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1017.35 | 988.27 | 983.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1019.35 | 1022.70 | 1010.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:45:00 | 1018.20 | 1022.70 | 1010.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1011.00 | 1020.36 | 1010.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 1011.00 | 1020.36 | 1010.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1019.50 | 1020.19 | 1011.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 1013.00 | 1020.19 | 1011.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1013.30 | 1018.81 | 1011.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 1013.30 | 1018.81 | 1011.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1014.60 | 1017.97 | 1011.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 1026.80 | 1015.78 | 1012.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1024.00 | 1018.12 | 1014.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 1008.65 | 1015.07 | 1014.22 | SL hit (close<static) qty=1.00 sl=1009.15 alert=retest2 |

### Cycle 170 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 1004.55 | 1012.96 | 1013.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 997.85 | 1008.09 | 1010.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 988.90 | 983.34 | 989.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 985.45 | 983.76 | 988.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1002.25 | 983.76 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 999.15 | 986.84 | 989.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 986.90 | 989.93 | 990.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 986.50 | 989.93 | 990.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 988.05 | 989.01 | 989.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 1001.95 | 994.79 | 992.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1000.40 | 1006.60 | 1001.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 994.00 | 1004.08 | 1000.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 1002.50 | 1003.55 | 1000.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-08 09:15:00 | 1102.75 | 1060.44 | 1045.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1073.20 | 1077.46 | 1077.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 1066.55 | 1075.53 | 1076.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1099.65 | 1070.17 | 1071.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 1094.95 | 1075.13 | 1073.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 1121.60 | 1094.25 | 1088.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 1121.90 | 1126.48 | 1117.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:30:00 | 1120.60 | 1126.48 | 1117.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1118.60 | 1123.99 | 1117.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 1115.00 | 1123.99 | 1117.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1108.15 | 1120.83 | 1116.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 1108.15 | 1120.83 | 1116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1111.70 | 1119.00 | 1116.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 1116.45 | 1116.74 | 1115.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1122.65 | 1125.96 | 1121.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1105.75 | 1121.05 | 1121.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 1105.75 | 1121.05 | 1121.71 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1119.20 | 1118.61 | 1118.54 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1116.60 | 1118.13 | 1118.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 1113.20 | 1117.14 | 1117.86 | Break + close below crossover candle low |

### Cycle 177 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1135.00 | 1119.17 | 1118.56 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1113.70 | 1121.95 | 1122.02 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 1126.90 | 1122.94 | 1122.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 1132.50 | 1124.85 | 1123.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1116.60 | 1128.80 | 1126.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1120.90 | 1127.22 | 1125.58 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 1120.00 | 1124.24 | 1124.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1116.40 | 1122.20 | 1123.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1133.10 | 1122.74 | 1123.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1141.50 | 1126.49 | 1124.99 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1118.20 | 1125.63 | 1126.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1111.40 | 1122.79 | 1125.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA400 retest candle locked (from downside) |

### Cycle 183 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1140.90 | 1126.39 | 1126.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 1144.50 | 1130.01 | 1127.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 1130.70 | 1133.68 | 1131.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1122.20 | 1131.38 | 1130.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 1122.20 | 1131.38 | 1130.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1122.70 | 1129.65 | 1129.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 1121.00 | 1127.92 | 1128.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1133.50 | 1121.28 | 1123.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1130.00 | 1123.03 | 1124.09 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1139.70 | 1126.36 | 1125.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1146.50 | 1134.98 | 1130.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA400 retest candle locked (from upside) |

### Cycle 186 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1126.40 | 1131.84 | 1132.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1121.50 | 1128.75 | 1130.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 10:15:00 | 1114.00 | 1110.85 | 1115.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:30:00 | 1114.90 | 1110.85 | 1115.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1110.30 | 1110.74 | 1115.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:45:00 | 1104.80 | 1110.63 | 1113.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:30:00 | 1106.10 | 1110.45 | 1113.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1088.60 | 1111.14 | 1112.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 12:15:00 | 1104.20 | 1111.35 | 1111.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1110.10 | 1107.76 | 1109.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1110.10 | 1107.76 | 1109.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1100.00 | 1106.20 | 1108.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1110.60 | 1106.94 | 1109.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1107.00 | 1106.95 | 1108.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1103.00 | 1106.06 | 1108.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 1100.00 | 1105.05 | 1107.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1125.60 | 1099.56 | 1094.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 1108.80 | 1109.10 | 1103.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 1108.70 | 1109.10 | 1103.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1102.30 | 1107.74 | 1103.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 1102.10 | 1107.74 | 1103.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1102.60 | 1106.71 | 1103.70 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1095.80 | 1101.97 | 1102.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1090.00 | 1099.58 | 1101.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1104.80 | 1099.83 | 1100.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1094.90 | 1098.84 | 1100.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 1094.20 | 1097.16 | 1099.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:00:00 | 1093.50 | 1096.43 | 1098.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1105.80 | 1097.27 | 1098.69 | SL hit (close>static) qty=1.00 sl=1105.10 alert=retest2 |

### Cycle 189 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 1100.00 | 1095.67 | 1095.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 12:15:00 | 1108.40 | 1100.30 | 1098.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 14:15:00 | 1102.00 | 1102.08 | 1099.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:45:00 | 1100.00 | 1102.08 | 1099.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1108.50 | 1103.36 | 1100.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1110.20 | 1104.73 | 1101.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:30:00 | 1110.50 | 1105.18 | 1101.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 1099.10 | 1103.56 | 1101.89 | SL hit (close<static) qty=1.00 sl=1099.90 alert=retest2 |

### Cycle 190 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 1095.10 | 1101.20 | 1101.97 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 1107.60 | 1102.43 | 1102.32 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1083.00 | 1099.69 | 1101.17 | EMA200 below EMA400 |

### Cycle 193 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 1116.60 | 1101.56 | 1101.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 1123.60 | 1105.97 | 1103.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 1151.80 | 1153.78 | 1142.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:30:00 | 1159.00 | 1153.78 | 1142.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1145.10 | 1150.89 | 1143.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1145.30 | 1150.89 | 1143.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1212.30 | 1228.34 | 1216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 1214.10 | 1228.34 | 1216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1214.70 | 1225.61 | 1216.35 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1196.50 | 1212.20 | 1213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1190.10 | 1207.78 | 1210.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 1205.60 | 1205.62 | 1209.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1205.00 | 1205.50 | 1208.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 1190.20 | 1201.28 | 1206.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1130.69 | 1146.91 | 1164.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 1148.60 | 1146.29 | 1161.15 | SL hit (close>ema200) qty=0.50 sl=1146.29 alert=retest2 |

### Cycle 195 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1119.90 | 1107.38 | 1105.79 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1103.00 | 1108.62 | 1108.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 1099.20 | 1106.47 | 1107.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1067.70 | 1058.26 | 1069.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 1067.70 | 1058.26 | 1069.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1068.50 | 1060.31 | 1069.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 1067.00 | 1060.31 | 1069.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1073.10 | 1062.87 | 1069.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1073.10 | 1062.87 | 1069.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1070.30 | 1064.35 | 1069.96 | EMA400 retest candle locked (from downside) |

### Cycle 197 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1081.50 | 1073.15 | 1072.97 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1069.20 | 1072.36 | 1072.63 | EMA200 below EMA400 |

### Cycle 199 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1078.60 | 1071.69 | 1071.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1096.00 | 1077.66 | 1074.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1096.60 | 1118.00 | 1107.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1100.10 | 1114.42 | 1107.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 1103.80 | 1109.61 | 1105.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 1103.30 | 1108.21 | 1105.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 1090.10 | 1103.59 | 1103.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1090.10 | 1103.59 | 1103.95 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 1108.90 | 1099.96 | 1099.95 | EMA200 above EMA400 |

### Cycle 202 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1090.70 | 1098.11 | 1099.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1088.00 | 1095.60 | 1097.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1091.50 | 1086.23 | 1091.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1078.50 | 1084.68 | 1090.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1072.00 | 1080.74 | 1086.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 1119.40 | 1073.03 | 1068.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 203 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1119.40 | 1073.03 | 1068.50 | EMA200 above EMA400 |

### Cycle 204 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1070.00 | 1075.39 | 1076.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1069.00 | 1074.12 | 1075.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1057.10 | 1073.17 | 1074.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1056.90 | 1055.44 | 1058.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:15:00 | 1004.24 | 1018.16 | 1028.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:15:00 | 1004.06 | 1018.16 | 1028.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1018.80 | 1007.17 | 1013.70 | SL hit (close>ema200) qty=0.50 sl=1007.17 alert=retest2 |

### Cycle 205 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1012.60 | 1007.28 | 1007.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1019.50 | 1010.96 | 1008.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1033.20 | 1018.41 | 1014.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1020.80 | 1031.95 | 1033.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1020.80 | 1031.95 | 1033.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1019.70 | 1029.50 | 1031.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA400 retest candle locked (from downside) |

### Cycle 207 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 1020.00 | 1005.72 | 1004.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 1047.80 | 1014.13 | 1008.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 12:15:00 | 1019.00 | 1024.29 | 1017.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:45:00 | 1019.40 | 1024.29 | 1017.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1021.30 | 1023.69 | 1017.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 1016.20 | 1023.69 | 1017.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1021.40 | 1023.24 | 1017.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 1019.10 | 1023.24 | 1017.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1017.00 | 1021.99 | 1017.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1013.90 | 1021.99 | 1017.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1014.00 | 1020.39 | 1017.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 1009.10 | 1020.39 | 1017.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1024.10 | 1019.49 | 1017.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 1029.60 | 1021.47 | 1018.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1009.70 | 1018.96 | 1018.18 | SL hit (close<static) qty=1.00 sl=1010.10 alert=retest2 |

### Cycle 208 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1009.20 | 1017.01 | 1017.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 996.90 | 1012.99 | 1015.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 973.80 | 957.18 | 969.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 963.30 | 958.41 | 968.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 954.30 | 959.03 | 968.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 945.30 | 940.93 | 940.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 945.30 | 940.93 | 940.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 952.80 | 943.30 | 941.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 967.50 | 968.53 | 959.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:30:00 | 969.95 | 968.53 | 959.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 991.05 | 973.04 | 962.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 978.60 | 973.04 | 962.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 953.30 | 969.09 | 961.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 953.30 | 969.09 | 961.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 959.00 | 967.07 | 961.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 960.70 | 967.07 | 961.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 959.00 | 965.93 | 962.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 960.50 | 965.93 | 962.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 964.00 | 965.54 | 962.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 967.70 | 965.97 | 963.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 980.50 | 965.78 | 963.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 984.90 | 992.68 | 993.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 984.90 | 992.68 | 993.26 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 999.85 | 994.21 | 993.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1007.50 | 996.87 | 995.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1072.45 | 1088.60 | 1072.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1079.40 | 1086.76 | 1072.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 1071.50 | 1086.76 | 1072.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1075.50 | 1084.51 | 1073.18 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 1057.80 | 1069.53 | 1069.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1039.80 | 1057.35 | 1063.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 1038.40 | 1037.69 | 1047.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:45:00 | 1039.15 | 1037.69 | 1047.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1049.00 | 1039.95 | 1047.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1049.00 | 1039.95 | 1047.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1038.90 | 1039.74 | 1046.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1031.60 | 1039.23 | 1044.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 1051.65 | 1045.34 | 1045.38 | SL hit (close>static) qty=1.00 sl=1050.95 alert=retest2 |

### Cycle 213 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1046.80 | 1045.63 | 1045.51 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1037.50 | 1044.01 | 1044.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1031.70 | 1041.55 | 1043.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1029.75 | 1028.15 | 1034.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:45:00 | 1030.75 | 1028.15 | 1034.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1032.00 | 1030.11 | 1033.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1032.00 | 1030.11 | 1033.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1031.50 | 1030.39 | 1033.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 1033.40 | 1030.39 | 1033.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1039.60 | 1032.23 | 1034.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1039.40 | 1032.23 | 1034.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1025.75 | 1030.94 | 1033.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1023.20 | 1029.80 | 1032.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1021.65 | 1029.80 | 1032.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 1021.75 | 1026.70 | 1030.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1017.65 | 1026.87 | 1029.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1012.85 | 1024.07 | 1028.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1023.60 | 1029.63 | 1029.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1022.00 | 1025.99 | 1027.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 1025.90 | 1025.02 | 1026.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 1015.30 | 1023.08 | 1025.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:30:00 | 1011.90 | 1020.10 | 1024.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 999.60 | 1018.02 | 1021.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 1033.20 | 1016.97 | 1016.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 1033.20 | 1016.97 | 1016.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 1040.00 | 1021.58 | 1018.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 1024.90 | 1024.98 | 1021.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:00:00 | 1024.90 | 1024.98 | 1021.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1023.20 | 1024.62 | 1021.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1022.80 | 1024.62 | 1021.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1022.90 | 1024.02 | 1021.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 1023.10 | 1024.02 | 1021.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1022.60 | 1023.73 | 1021.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 1022.60 | 1023.73 | 1021.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1023.70 | 1023.73 | 1021.96 | EMA400 retest candle locked (from upside) |

### Cycle 218 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 991.10 | 1017.04 | 1019.22 | EMA200 below EMA400 |

### Cycle 219 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 1036.00 | 1018.15 | 1017.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 10:15:00 | 1040.00 | 1022.52 | 1019.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 1035.30 | 1039.59 | 1031.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1035.00 | 1037.41 | 1032.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1034.60 | 1037.41 | 1032.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1026.00 | 1035.13 | 1031.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1026.00 | 1035.13 | 1031.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1022.00 | 1032.50 | 1030.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1003.70 | 1032.50 | 1030.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1005.70 | 1027.14 | 1028.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 995.00 | 1012.62 | 1020.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 978.10 | 972.67 | 985.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 973.90 | 972.67 | 985.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 988.30 | 976.66 | 985.11 | EMA400 retest candle locked (from downside) |

### Cycle 221 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1002.00 | 989.32 | 989.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1008.00 | 993.06 | 990.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1012.20 | 1016.46 | 1008.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 995.50 | 1012.26 | 1007.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 995.50 | 1012.26 | 1007.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1001.00 | 1010.01 | 1006.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1027.70 | 1005.92 | 1005.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:30:00 | 1009.00 | 1006.77 | 1006.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 1008.40 | 1007.16 | 1006.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 985.80 | 1000.70 | 1003.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 938.90 | 955.11 | 959.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 946.35 | 938.40 | 938.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 946.35 | 938.40 | 938.35 | EMA200 above EMA400 |

### Cycle 224 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 923.90 | 936.94 | 937.86 | EMA200 below EMA400 |

### Cycle 225 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 945.05 | 938.83 | 938.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 953.85 | 942.54 | 940.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 965.00 | 946.62 | 943.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 13:15:00 | 942.90 | 952.82 | 952.09 | SL hit (close<static) qty=1.00 sl=943.20 alert=retest2 |

### Cycle 226 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 942.90 | 950.84 | 951.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 933.85 | 940.56 | 944.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 960.00 | 942.14 | 944.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 951.40 | 943.99 | 945.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:15:00 | 945.25 | 943.99 | 945.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 951.25 | 946.33 | 945.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 951.25 | 946.33 | 945.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 953.95 | 949.27 | 947.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 945.60 | 948.54 | 947.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 939.45 | 946.72 | 946.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 939.45 | 946.72 | 946.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 939.80 | 945.34 | 945.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 14:15:00 | 936.35 | 942.66 | 944.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 941.60 | 938.19 | 941.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 229 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 968.20 | 944.08 | 942.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 976.15 | 967.45 | 958.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 15:15:00 | 999.00 | 1003.18 | 995.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1014.55 | 1003.18 | 995.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 1065.28 | 1030.90 | 1015.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1068.00 | 1075.37 | 1057.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1068.00 | 1075.37 | 1057.67 | SL hit (close<ema200) qty=0.50 sl=1075.37 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-15 13:15:00 | 1397.50 | 2024-04-15 14:15:00 | 1396.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-04-26 10:15:00 | 1402.55 | 2024-05-03 09:15:00 | 1332.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 13:30:00 | 1403.10 | 2024-05-03 09:15:00 | 1332.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 14:00:00 | 1402.80 | 2024-05-03 09:15:00 | 1332.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-26 10:15:00 | 1402.55 | 2024-05-06 13:15:00 | 1313.00 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2024-04-26 13:30:00 | 1403.10 | 2024-05-06 13:15:00 | 1313.00 | STOP_HIT | 0.50 | 6.42% |
| SELL | retest2 | 2024-04-26 14:00:00 | 1402.80 | 2024-05-06 13:15:00 | 1313.00 | STOP_HIT | 0.50 | 6.40% |
| BUY | retest2 | 2024-05-23 14:00:00 | 1350.75 | 2024-05-27 09:15:00 | 1334.05 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-05-23 15:00:00 | 1349.10 | 2024-05-27 09:15:00 | 1334.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-05-30 09:15:00 | 1322.80 | 2024-06-03 12:15:00 | 1334.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-03 10:00:00 | 1328.40 | 2024-06-03 12:15:00 | 1334.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-06-10 10:15:00 | 1350.90 | 2024-06-20 12:15:00 | 1384.75 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2024-06-10 14:45:00 | 1348.05 | 2024-06-20 12:15:00 | 1384.75 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2024-06-11 09:15:00 | 1361.20 | 2024-06-20 12:15:00 | 1384.75 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2024-06-25 10:30:00 | 1459.10 | 2024-06-28 14:15:00 | 1426.45 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2024-06-26 09:30:00 | 1444.70 | 2024-06-28 14:15:00 | 1426.45 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-07-09 13:00:00 | 1458.10 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-07-09 14:15:00 | 1458.60 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-07-09 14:45:00 | 1454.55 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-07-09 15:15:00 | 1456.35 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-07-10 10:15:00 | 1429.05 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-07-10 12:15:00 | 1438.45 | 2024-07-11 11:15:00 | 1470.90 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-07-22 14:00:00 | 1425.80 | 2024-07-24 09:15:00 | 1469.45 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2024-07-22 15:15:00 | 1426.00 | 2024-07-24 09:15:00 | 1469.45 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1474.55 | 2024-08-13 13:15:00 | 1462.70 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-08-20 14:00:00 | 1511.50 | 2024-08-23 10:15:00 | 1497.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1516.90 | 2024-08-23 10:15:00 | 1497.70 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-09-09 14:45:00 | 1563.05 | 2024-09-19 11:15:00 | 1647.80 | STOP_HIT | 1.00 | 5.42% |
| BUY | retest2 | 2024-09-26 11:15:00 | 1701.85 | 2024-09-27 14:15:00 | 1686.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-09-27 13:45:00 | 1703.80 | 2024-09-27 14:15:00 | 1686.30 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-10-08 12:15:00 | 1566.10 | 2024-10-08 13:15:00 | 1605.60 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-10-10 13:15:00 | 1602.45 | 2024-10-14 10:15:00 | 1592.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-10-10 13:45:00 | 1604.50 | 2024-10-14 10:15:00 | 1592.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-10-10 14:45:00 | 1605.05 | 2024-10-14 10:15:00 | 1592.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-11-12 11:15:00 | 1488.20 | 2024-11-19 11:15:00 | 1487.30 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-01-15 13:30:00 | 1081.90 | 2025-01-23 14:15:00 | 1093.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-15 14:00:00 | 1082.35 | 2025-01-23 14:15:00 | 1093.70 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-01-16 14:45:00 | 1081.95 | 2025-01-23 14:15:00 | 1093.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-01-21 14:30:00 | 1077.50 | 2025-01-23 14:15:00 | 1093.70 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-01-31 09:15:00 | 1072.50 | 2025-02-07 09:15:00 | 1077.85 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2025-02-01 12:00:00 | 1078.40 | 2025-02-07 09:15:00 | 1077.85 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-02-12 13:45:00 | 1046.65 | 2025-02-14 13:15:00 | 1007.43 | PARTIAL | 0.50 | 3.75% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1056.50 | 2025-02-14 13:15:00 | 1007.47 | PARTIAL | 0.50 | 4.64% |
| SELL | retest2 | 2025-02-13 11:30:00 | 1060.45 | 2025-02-17 09:15:00 | 1003.67 | PARTIAL | 0.50 | 5.35% |
| SELL | retest2 | 2025-02-13 14:30:00 | 1060.50 | 2025-02-18 10:15:00 | 994.32 | PARTIAL | 0.50 | 6.24% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1040.90 | 2025-02-18 12:15:00 | 988.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-12 13:45:00 | 1046.65 | 2025-02-19 14:15:00 | 990.25 | STOP_HIT | 0.50 | 5.39% |
| SELL | retest2 | 2025-02-13 10:15:00 | 1056.50 | 2025-02-19 14:15:00 | 990.25 | STOP_HIT | 0.50 | 6.27% |
| SELL | retest2 | 2025-02-13 11:30:00 | 1060.45 | 2025-02-19 14:15:00 | 990.25 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2025-02-13 14:30:00 | 1060.50 | 2025-02-19 14:15:00 | 990.25 | STOP_HIT | 0.50 | 6.62% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1040.90 | 2025-02-19 14:15:00 | 990.25 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-03-12 09:45:00 | 932.00 | 2025-03-17 12:15:00 | 885.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 09:45:00 | 932.00 | 2025-03-17 14:15:00 | 900.65 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2025-03-18 10:30:00 | 934.60 | 2025-03-18 11:15:00 | 934.95 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-03-21 09:45:00 | 963.00 | 2025-03-25 11:15:00 | 950.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-25 09:45:00 | 964.15 | 2025-03-25 11:15:00 | 950.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-03-28 11:30:00 | 918.90 | 2025-04-01 10:15:00 | 955.25 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-04-09 09:15:00 | 873.10 | 2025-04-11 09:15:00 | 916.80 | STOP_HIT | 1.00 | -5.01% |
| BUY | retest1 | 2025-04-17 12:45:00 | 953.50 | 2025-04-24 11:15:00 | 1001.18 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-17 14:00:00 | 953.50 | 2025-04-24 11:15:00 | 1001.18 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 12:15:00 | 953.10 | 2025-04-24 11:15:00 | 1000.76 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-21 13:00:00 | 955.55 | 2025-04-24 12:15:00 | 1003.33 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-04-17 12:45:00 | 953.50 | 2025-04-25 09:15:00 | 984.15 | STOP_HIT | 0.50 | 3.21% |
| BUY | retest1 | 2025-04-17 14:00:00 | 953.50 | 2025-04-25 09:15:00 | 984.15 | STOP_HIT | 0.50 | 3.21% |
| BUY | retest1 | 2025-04-21 12:15:00 | 953.10 | 2025-04-25 09:15:00 | 984.15 | STOP_HIT | 0.50 | 3.26% |
| BUY | retest1 | 2025-04-21 13:00:00 | 955.55 | 2025-04-25 09:15:00 | 984.15 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-05-05 09:15:00 | 947.55 | 2025-05-08 11:15:00 | 949.05 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-05-06 09:15:00 | 946.05 | 2025-05-08 11:15:00 | 949.05 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-05-15 12:30:00 | 986.60 | 2025-05-20 10:15:00 | 969.85 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-20 10:15:00 | 969.55 | 2025-05-20 10:15:00 | 969.85 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1008.85 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-29 15:15:00 | 1003.85 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-05-30 10:45:00 | 1001.60 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-30 12:45:00 | 1001.55 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1008.00 | 2025-06-13 09:15:00 | 957.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1008.00 | 2025-06-16 13:15:00 | 958.25 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-07-15 11:15:00 | 987.80 | 2025-07-15 11:15:00 | 1005.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-23 12:15:00 | 1026.80 | 2025-07-24 11:15:00 | 1008.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1024.00 | 2025-07-24 11:15:00 | 1008.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-30 14:30:00 | 986.90 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-30 15:15:00 | 986.50 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-31 10:00:00 | 988.05 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-04 10:45:00 | 1002.50 | 2025-08-08 09:15:00 | 1102.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 09:30:00 | 1116.45 | 2025-08-29 09:15:00 | 1105.75 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-28 09:15:00 | 1122.65 | 2025-08-29 09:15:00 | 1105.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-24 11:45:00 | 1104.80 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-24 12:30:00 | 1106.10 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1088.60 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-26 12:15:00 | 1104.20 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1103.00 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-29 15:00:00 | 1100.00 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-08 13:45:00 | 1094.20 | 2025-10-09 09:15:00 | 1105.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-08 15:00:00 | 1093.50 | 2025-10-09 09:15:00 | 1105.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-09 13:00:00 | 1094.40 | 2025-10-13 09:15:00 | 1098.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-10 11:45:00 | 1094.60 | 2025-10-13 09:15:00 | 1098.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-10 13:30:00 | 1091.90 | 2025-10-13 11:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-10-10 14:00:00 | 1091.90 | 2025-10-13 11:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1110.20 | 2025-10-15 13:15:00 | 1099.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-15 10:30:00 | 1110.50 | 2025-10-15 13:15:00 | 1099.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1111.80 | 2025-10-16 14:15:00 | 1096.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-11-04 09:30:00 | 1190.20 | 2025-11-07 09:15:00 | 1130.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:30:00 | 1190.20 | 2025-11-07 11:15:00 | 1148.60 | STOP_HIT | 0.50 | 3.50% |
| BUY | retest2 | 2025-12-04 12:45:00 | 1103.80 | 2025-12-04 15:15:00 | 1090.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-04 14:15:00 | 1103.30 | 2025-12-04 15:15:00 | 1090.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1072.00 | 2025-12-15 10:15:00 | 1119.40 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1057.10 | 2025-12-26 11:15:00 | 1004.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1056.90 | 2025-12-26 11:15:00 | 1004.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1057.10 | 2025-12-29 14:15:00 | 1018.80 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1056.90 | 2025-12-29 14:15:00 | 1018.80 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2026-01-02 09:15:00 | 1033.20 | 2026-01-08 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-16 15:00:00 | 1029.60 | 2026-01-19 09:15:00 | 1009.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-22 12:15:00 | 954.30 | 2026-01-29 15:15:00 | 945.30 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2026-02-02 15:00:00 | 967.70 | 2026-02-06 10:15:00 | 984.90 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2026-02-03 09:15:00 | 980.50 | 2026-02-06 10:15:00 | 984.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-02-18 09:30:00 | 1031.60 | 2026-02-18 15:15:00 | 1051.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-23 10:30:00 | 1023.20 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1021.65 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-23 13:45:00 | 1021.75 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1017.65 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-03-02 12:30:00 | 1011.90 | 2026-03-05 11:15:00 | 1033.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-03-04 09:15:00 | 999.60 | 2026-03-05 11:15:00 | 1033.20 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1027.70 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-20 11:30:00 | 1009.00 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-20 12:45:00 | 1008.40 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-03-27 09:15:00 | 938.90 | 2026-04-01 13:15:00 | 946.35 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-04-08 09:15:00 | 965.00 | 2026-04-09 13:15:00 | 942.90 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-15 11:15:00 | 945.25 | 2026-04-16 10:15:00 | 951.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-04-27 09:15:00 | 1014.55 | 2026-04-28 10:15:00 | 1065.28 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-27 09:15:00 | 1014.55 | 2026-04-30 09:15:00 | 1068.00 | STOP_HIT | 0.50 | 5.27% |
