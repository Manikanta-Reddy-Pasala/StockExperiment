# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1419.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 251 |
| ALERT1 | 149 |
| ALERT2 | 146 |
| ALERT2_SKIP | 112 |
| ALERT3 | 309 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 128 |
| PARTIAL | 15 |
| TARGET_HIT | 1 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 144 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 58 / 86
- **Target hits / Stop hits / Partials:** 1 / 128 / 15
- **Avg / median % per leg:** 0.45% / -0.40%
- **Sum % (uncompounded):** 64.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 8 | 14.3% | 1 | 55 | 0 | -0.73% | -40.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 56 | 8 | 14.3% | 1 | 55 | 0 | -0.73% | -40.9% |
| SELL (all) | 88 | 50 | 56.8% | 0 | 73 | 15 | 1.19% | 105.0% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.16% | 1.2% |
| SELL @ 3rd Alert (retest2) | 87 | 49 | 56.3% | 0 | 72 | 15 | 1.19% | 103.8% |
| retest1 (combined) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.16% | 1.2% |
| retest2 (combined) | 143 | 57 | 39.9% | 1 | 127 | 15 | 0.44% | 63.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 11:15:00 | 1428.00 | 1429.58 | 1429.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 1410.00 | 1423.73 | 1426.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 1416.25 | 1414.15 | 1420.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 1407.90 | 1413.36 | 1418.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 1407.90 | 1413.36 | 1418.67 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 1420.10 | 1413.03 | 1412.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 15:15:00 | 1425.00 | 1418.00 | 1415.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 15:15:00 | 1455.00 | 1455.65 | 1448.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 14:15:00 | 1450.00 | 1453.54 | 1450.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 14:15:00 | 1450.00 | 1453.54 | 1450.49 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 11:15:00 | 1446.50 | 1456.45 | 1456.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 09:15:00 | 1444.85 | 1448.28 | 1450.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 14:15:00 | 1441.95 | 1441.35 | 1445.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 15:15:00 | 1446.00 | 1442.28 | 1445.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 15:15:00 | 1446.00 | 1442.28 | 1445.83 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 09:15:00 | 1473.00 | 1448.43 | 1448.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 1497.00 | 1458.14 | 1452.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-08 11:15:00 | 1486.15 | 1491.05 | 1477.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 14:15:00 | 1481.50 | 1487.52 | 1479.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 1481.50 | 1487.52 | 1479.10 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 11:15:00 | 1475.00 | 1478.36 | 1478.59 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 15:15:00 | 1483.15 | 1478.88 | 1478.65 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 14:15:00 | 1465.45 | 1476.94 | 1478.32 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-14 10:15:00 | 1490.25 | 1480.94 | 1479.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 09:15:00 | 1495.55 | 1489.59 | 1486.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 10:15:00 | 1495.25 | 1505.71 | 1499.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-19 10:15:00 | 1495.25 | 1505.71 | 1499.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 10:15:00 | 1495.25 | 1505.71 | 1499.11 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 1489.50 | 1495.61 | 1496.39 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 1499.75 | 1496.66 | 1496.28 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 1486.90 | 1494.71 | 1495.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 15:15:00 | 1485.00 | 1489.79 | 1492.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-22 10:15:00 | 1491.70 | 1488.94 | 1491.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 1491.70 | 1488.94 | 1491.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1491.70 | 1488.94 | 1491.55 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 13:15:00 | 1508.95 | 1494.60 | 1493.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 14:15:00 | 1510.10 | 1497.70 | 1495.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 09:15:00 | 1484.75 | 1496.24 | 1494.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 09:15:00 | 1484.75 | 1496.24 | 1494.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 1484.75 | 1496.24 | 1494.93 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 11:15:00 | 1490.60 | 1494.11 | 1494.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 09:15:00 | 1486.30 | 1490.76 | 1492.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 11:15:00 | 1492.90 | 1490.84 | 1492.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 11:15:00 | 1492.90 | 1490.84 | 1492.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 1492.90 | 1490.84 | 1492.07 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 14:15:00 | 1498.70 | 1493.08 | 1492.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 13:15:00 | 1502.45 | 1497.52 | 1496.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 13:15:00 | 1505.00 | 1506.92 | 1502.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 15:15:00 | 1517.00 | 1516.88 | 1513.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 1517.00 | 1516.88 | 1513.09 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 1509.95 | 1526.11 | 1526.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 15:15:00 | 1497.00 | 1517.60 | 1522.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1489.15 | 1488.86 | 1500.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 1491.45 | 1488.22 | 1494.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1491.45 | 1488.22 | 1494.79 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 11:15:00 | 1502.20 | 1492.31 | 1492.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 1510.75 | 1501.95 | 1498.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 09:15:00 | 1507.05 | 1512.13 | 1507.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 09:15:00 | 1507.05 | 1512.13 | 1507.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 09:15:00 | 1507.05 | 1512.13 | 1507.66 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 1498.45 | 1504.61 | 1504.91 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 11:15:00 | 1512.50 | 1504.28 | 1503.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 1518.55 | 1507.12 | 1505.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 13:15:00 | 1517.90 | 1517.96 | 1512.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 09:15:00 | 1498.90 | 1515.77 | 1512.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1498.90 | 1515.77 | 1512.62 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 11:15:00 | 1491.45 | 1507.42 | 1509.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 10:15:00 | 1482.00 | 1496.22 | 1502.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 11:15:00 | 1483.50 | 1483.29 | 1490.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 13:15:00 | 1481.55 | 1483.74 | 1489.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 13:15:00 | 1481.55 | 1483.74 | 1489.56 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-07-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 09:15:00 | 1520.05 | 1495.69 | 1493.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 10:15:00 | 1532.00 | 1502.95 | 1497.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 09:15:00 | 1547.60 | 1556.71 | 1540.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 1547.60 | 1556.71 | 1540.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 1547.60 | 1556.71 | 1540.70 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 09:15:00 | 1550.00 | 1560.25 | 1560.28 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 10:15:00 | 1577.90 | 1561.21 | 1559.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 1581.40 | 1569.78 | 1564.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 14:15:00 | 1585.10 | 1586.64 | 1577.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 1567.90 | 1582.15 | 1577.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 1567.90 | 1582.15 | 1577.18 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 12:15:00 | 1561.80 | 1572.65 | 1573.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-08 13:15:00 | 1558.00 | 1569.72 | 1572.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 10:15:00 | 1541.15 | 1540.97 | 1548.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-14 14:15:00 | 1539.00 | 1534.77 | 1539.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-14 14:15:00 | 1539.00 | 1534.77 | 1539.15 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 1552.00 | 1541.43 | 1540.15 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 09:15:00 | 1527.10 | 1540.45 | 1540.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 1517.20 | 1533.75 | 1537.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 11:15:00 | 1525.00 | 1520.60 | 1527.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 12:15:00 | 1535.00 | 1523.48 | 1528.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 12:15:00 | 1535.00 | 1523.48 | 1528.22 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 12:15:00 | 1535.50 | 1529.77 | 1529.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 15:15:00 | 1539.90 | 1532.56 | 1531.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 09:15:00 | 1529.15 | 1531.88 | 1530.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 09:15:00 | 1529.15 | 1531.88 | 1530.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 1529.15 | 1531.88 | 1530.84 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 12:15:00 | 1529.05 | 1531.11 | 1531.24 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 13:15:00 | 1534.60 | 1531.81 | 1531.54 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 1524.00 | 1530.63 | 1531.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 1513.25 | 1524.85 | 1528.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 11:15:00 | 1516.95 | 1515.70 | 1520.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 12:15:00 | 1518.60 | 1516.28 | 1520.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 1518.60 | 1516.28 | 1520.68 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 1529.25 | 1521.76 | 1521.73 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 13:15:00 | 1521.10 | 1521.60 | 1521.66 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 1526.25 | 1522.53 | 1522.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 15:15:00 | 1529.00 | 1523.82 | 1522.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 1529.35 | 1532.08 | 1528.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 1529.35 | 1532.08 | 1528.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1529.35 | 1532.08 | 1528.85 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 1516.85 | 1525.21 | 1526.30 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 14:15:00 | 1545.00 | 1528.32 | 1526.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 15:15:00 | 1551.75 | 1533.00 | 1528.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 09:15:00 | 1664.20 | 1668.44 | 1643.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 1639.55 | 1656.60 | 1648.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1639.55 | 1656.60 | 1648.83 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 12:15:00 | 1616.85 | 1639.01 | 1641.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 1611.00 | 1629.99 | 1637.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 1619.95 | 1610.89 | 1619.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 1619.95 | 1610.89 | 1619.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1619.95 | 1610.89 | 1619.95 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 1582.20 | 1577.09 | 1576.78 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 1572.00 | 1576.00 | 1576.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 1565.20 | 1573.84 | 1575.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 11:15:00 | 1553.95 | 1553.52 | 1562.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 13:15:00 | 1557.90 | 1555.38 | 1561.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 1557.90 | 1555.38 | 1561.88 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 11:15:00 | 1570.60 | 1563.99 | 1563.89 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-03 13:15:00 | 1559.20 | 1563.67 | 1563.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-03 15:15:00 | 1555.20 | 1561.16 | 1562.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 10:15:00 | 1565.00 | 1559.61 | 1561.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 10:15:00 | 1565.00 | 1559.61 | 1561.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 1565.00 | 1559.61 | 1561.49 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 10:15:00 | 1565.50 | 1556.89 | 1556.32 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 12:15:00 | 1554.30 | 1555.86 | 1555.92 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 13:15:00 | 1557.70 | 1556.22 | 1556.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 14:15:00 | 1560.95 | 1557.17 | 1556.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 10:15:00 | 1560.50 | 1562.20 | 1559.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 1560.50 | 1562.20 | 1559.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 1560.50 | 1562.20 | 1559.28 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 09:15:00 | 1552.80 | 1580.15 | 1583.09 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-20 09:15:00 | 1613.60 | 1574.87 | 1572.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-25 09:15:00 | 1647.60 | 1628.68 | 1612.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-25 11:15:00 | 1628.90 | 1631.13 | 1616.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 12:15:00 | 1607.20 | 1626.34 | 1615.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 12:15:00 | 1607.20 | 1626.34 | 1615.93 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 1574.25 | 1604.64 | 1608.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 10:15:00 | 1564.80 | 1596.67 | 1604.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1593.35 | 1580.29 | 1590.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1593.35 | 1580.29 | 1590.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1593.35 | 1580.29 | 1590.22 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 1608.90 | 1595.50 | 1595.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 10:15:00 | 1622.75 | 1609.18 | 1603.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 15:15:00 | 1612.00 | 1612.44 | 1607.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 12:15:00 | 1608.75 | 1612.72 | 1609.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 12:15:00 | 1608.75 | 1612.72 | 1609.56 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 11:15:00 | 1607.85 | 1610.74 | 1611.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 12:15:00 | 1603.25 | 1609.24 | 1610.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 12:15:00 | 1598.95 | 1595.47 | 1601.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 14:15:00 | 1598.40 | 1596.75 | 1600.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 14:15:00 | 1598.40 | 1596.75 | 1600.70 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 13:15:00 | 1575.90 | 1570.29 | 1569.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 14:15:00 | 1584.50 | 1573.13 | 1571.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 10:15:00 | 1575.60 | 1575.79 | 1573.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 11:15:00 | 1580.60 | 1576.75 | 1573.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 1580.60 | 1576.75 | 1573.71 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 15:15:00 | 1580.00 | 1594.75 | 1594.90 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 10:15:00 | 1599.90 | 1595.50 | 1595.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 11:15:00 | 1605.30 | 1597.46 | 1596.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 15:15:00 | 1705.00 | 1709.41 | 1699.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 1707.45 | 1709.01 | 1700.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1707.45 | 1709.01 | 1700.01 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2023-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 15:15:00 | 1716.80 | 1732.57 | 1733.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 09:15:00 | 1694.95 | 1725.04 | 1729.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 1710.50 | 1709.94 | 1718.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 1706.50 | 1708.49 | 1716.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1706.50 | 1708.49 | 1716.60 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 1744.60 | 1712.47 | 1710.09 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 1686.00 | 1707.70 | 1709.98 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 13:15:00 | 1709.25 | 1703.99 | 1703.42 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 12:15:00 | 1696.85 | 1703.73 | 1703.88 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 14:15:00 | 1711.35 | 1704.73 | 1704.27 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2023-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 15:15:00 | 1692.25 | 1702.23 | 1703.18 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2023-12-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 09:15:00 | 1722.65 | 1706.32 | 1704.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 1731.25 | 1711.30 | 1707.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 1785.90 | 1786.70 | 1767.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 09:15:00 | 1869.45 | 1882.37 | 1866.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 09:15:00 | 1869.45 | 1882.37 | 1866.26 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 09:15:00 | 1836.50 | 1862.43 | 1863.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 13:15:00 | 1823.50 | 1842.33 | 1852.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 13:15:00 | 1797.10 | 1795.95 | 1806.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 1805.95 | 1798.00 | 1804.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1805.95 | 1798.00 | 1804.51 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 12:15:00 | 1826.40 | 1809.11 | 1808.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 09:15:00 | 1831.15 | 1819.75 | 1814.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 10:15:00 | 1817.90 | 1819.38 | 1814.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 10:15:00 | 1817.90 | 1819.38 | 1814.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 1817.90 | 1819.38 | 1814.50 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 11:15:00 | 1813.10 | 1830.52 | 1830.89 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 1845.00 | 1832.93 | 1831.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 12:15:00 | 1852.25 | 1838.79 | 1834.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 1840.45 | 1841.11 | 1836.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 15:15:00 | 1842.90 | 1841.46 | 1837.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1842.90 | 1841.46 | 1837.25 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 09:15:00 | 1820.20 | 1845.94 | 1849.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 11:15:00 | 1813.40 | 1835.59 | 1843.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 09:15:00 | 1819.30 | 1818.98 | 1831.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 1811.40 | 1808.57 | 1813.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 1811.40 | 1808.57 | 1813.13 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 1841.40 | 1816.33 | 1812.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 14:15:00 | 1845.95 | 1827.85 | 1819.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 09:15:00 | 1822.15 | 1829.29 | 1821.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 1822.15 | 1829.29 | 1821.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 1822.15 | 1829.29 | 1821.77 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 12:15:00 | 1786.50 | 1811.57 | 1814.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 11:15:00 | 1771.95 | 1784.35 | 1791.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 09:15:00 | 1798.60 | 1779.01 | 1785.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 09:15:00 | 1798.60 | 1779.01 | 1785.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 1798.60 | 1779.01 | 1785.13 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-02-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 12:15:00 | 1764.80 | 1744.66 | 1743.02 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 13:15:00 | 1731.65 | 1745.97 | 1746.52 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 11:15:00 | 1757.75 | 1747.43 | 1746.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 12:15:00 | 1765.20 | 1750.98 | 1748.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1746.45 | 1756.65 | 1752.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 1746.45 | 1756.65 | 1752.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1746.45 | 1756.65 | 1752.46 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 1736.00 | 1749.06 | 1749.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 13:15:00 | 1726.95 | 1742.09 | 1746.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 11:15:00 | 1724.00 | 1723.49 | 1730.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 13:15:00 | 1728.85 | 1724.61 | 1730.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1728.85 | 1724.61 | 1730.07 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 1699.70 | 1689.34 | 1688.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 1709.55 | 1698.75 | 1694.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-02 12:15:00 | 1700.05 | 1700.69 | 1695.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 1697.00 | 1699.95 | 1696.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1697.00 | 1699.95 | 1696.03 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 1683.70 | 1701.01 | 1701.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 10:15:00 | 1676.05 | 1696.02 | 1699.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 1701.90 | 1694.86 | 1697.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 1701.90 | 1694.86 | 1697.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 1701.90 | 1694.86 | 1697.90 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 15:15:00 | 1718.90 | 1702.89 | 1701.22 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-07 15:15:00 | 1695.90 | 1702.47 | 1702.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-11 10:15:00 | 1680.15 | 1697.31 | 1700.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-11 14:15:00 | 1698.00 | 1696.28 | 1698.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 1698.00 | 1696.28 | 1698.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 1698.00 | 1696.28 | 1698.65 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 09:15:00 | 1733.50 | 1704.40 | 1701.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 12:15:00 | 1741.00 | 1714.36 | 1707.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 15:15:00 | 1711.15 | 1716.44 | 1710.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-12 15:15:00 | 1711.15 | 1716.44 | 1710.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 1711.15 | 1716.44 | 1710.30 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 14:15:00 | 1684.00 | 1709.72 | 1710.28 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 1723.65 | 1711.11 | 1710.46 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 13:15:00 | 1703.95 | 1709.68 | 1710.00 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-03-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 14:15:00 | 1720.30 | 1711.81 | 1710.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 15:15:00 | 1725.15 | 1714.48 | 1712.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 09:15:00 | 1700.00 | 1711.58 | 1711.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 1700.00 | 1711.58 | 1711.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1700.00 | 1711.58 | 1711.12 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 10:15:00 | 1700.30 | 1709.32 | 1710.14 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-03-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 13:15:00 | 1724.50 | 1710.82 | 1710.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 15:15:00 | 1741.05 | 1720.51 | 1715.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 1713.95 | 1719.20 | 1715.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 1713.95 | 1719.20 | 1715.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 1713.95 | 1719.20 | 1715.04 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 10:15:00 | 1691.75 | 1714.47 | 1715.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 11:15:00 | 1686.00 | 1708.78 | 1712.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 10:15:00 | 1689.75 | 1689.65 | 1699.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 1701.05 | 1691.28 | 1698.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1701.05 | 1691.28 | 1698.35 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-03-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 09:15:00 | 1722.35 | 1705.26 | 1703.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 10:15:00 | 1725.00 | 1716.33 | 1710.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 15:15:00 | 1718.20 | 1723.91 | 1717.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 1721.30 | 1723.39 | 1717.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1721.30 | 1723.39 | 1717.65 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2024-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 09:15:00 | 1709.10 | 1716.26 | 1717.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 11:15:00 | 1699.45 | 1711.26 | 1714.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 11:15:00 | 1704.05 | 1703.35 | 1707.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 11:15:00 | 1704.05 | 1703.35 | 1707.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 11:15:00 | 1704.05 | 1703.35 | 1707.96 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 14:15:00 | 1739.00 | 1714.93 | 1712.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 15:15:00 | 1744.60 | 1720.87 | 1715.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-03 14:15:00 | 1803.45 | 1807.07 | 1789.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 09:15:00 | 1779.80 | 1800.98 | 1789.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1779.80 | 1800.98 | 1789.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1839.90 | 1829.26 | 1823.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-19 09:15:00 | 1835.75 | 1845.82 | 1845.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-04-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-19 09:15:00 | 1835.75 | 1845.82 | 1845.94 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 1854.95 | 1845.42 | 1844.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 13:15:00 | 1856.75 | 1847.69 | 1845.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 09:15:00 | 1843.00 | 1848.17 | 1846.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 09:15:00 | 1843.00 | 1848.17 | 1846.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1843.00 | 1848.17 | 1846.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 09:45:00 | 1843.65 | 1848.17 | 1846.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1850.65 | 1848.67 | 1846.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 14:30:00 | 1854.45 | 1851.72 | 1848.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-24 15:15:00 | 2039.90 | 1949.00 | 1902.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 2019.75 | 2038.85 | 2039.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 12:15:00 | 1978.75 | 2011.40 | 2024.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 09:15:00 | 2005.30 | 2002.10 | 2015.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 2005.30 | 2002.10 | 2015.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 2005.30 | 2002.10 | 2015.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:45:00 | 2011.75 | 2002.10 | 2015.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 2020.00 | 1996.18 | 2004.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 1981.90 | 1996.18 | 2004.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 09:15:00 | 1961.50 | 1997.24 | 1999.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 13:15:00 | 1882.81 | 1904.08 | 1916.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 1899.40 | 1896.40 | 1904.54 | SL hit (close>ema200) qty=0.50 sl=1896.40 alert=retest2 |

### Cycle 88 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 1921.60 | 1905.87 | 1905.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 1925.00 | 1914.09 | 1909.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1877.85 | 1908.43 | 1908.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1877.85 | 1908.43 | 1908.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1877.85 | 1908.43 | 1908.01 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 1876.80 | 1902.10 | 1905.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 1860.80 | 1880.03 | 1890.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 1875.40 | 1871.05 | 1882.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 1875.40 | 1871.05 | 1882.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1875.40 | 1871.05 | 1882.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1875.40 | 1871.05 | 1882.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1880.90 | 1873.34 | 1881.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 1870.50 | 1875.15 | 1881.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 1870.75 | 1878.96 | 1881.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 11:00:00 | 1872.15 | 1874.73 | 1879.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 15:00:00 | 1874.05 | 1872.84 | 1876.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1875.00 | 1873.14 | 1876.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:15:00 | 1860.25 | 1871.95 | 1874.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:00:00 | 1863.35 | 1870.23 | 1873.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:30:00 | 1860.30 | 1869.94 | 1872.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:30:00 | 1862.40 | 1869.94 | 1872.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 1873.65 | 1870.68 | 1872.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:15:00 | 1869.75 | 1870.68 | 1872.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 1866.10 | 1869.77 | 1871.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:15:00 | 1860.15 | 1870.61 | 1872.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:45:00 | 1864.15 | 1870.32 | 1871.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 12:15:00 | 1877.95 | 1871.85 | 1872.07 | SL hit (close>static) qty=1.00 sl=1874.85 alert=retest2 |

### Cycle 90 — BUY (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 14:15:00 | 1877.90 | 1872.90 | 1872.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-29 15:15:00 | 1880.00 | 1874.32 | 1873.18 | Break + close above crossover candle high |

### Cycle 91 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1842.15 | 1867.88 | 1870.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 1836.35 | 1861.58 | 1867.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1858.85 | 1849.59 | 1856.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 1858.85 | 1849.59 | 1856.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1858.85 | 1849.59 | 1856.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 1858.85 | 1849.59 | 1856.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1851.90 | 1850.05 | 1856.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:15:00 | 1848.90 | 1850.05 | 1856.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1865.95 | 1857.94 | 1857.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 1865.95 | 1857.94 | 1857.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 1955.60 | 1885.11 | 1873.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 2139.90 | 2141.15 | 2110.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 2139.90 | 2141.15 | 2110.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 2117.45 | 2133.45 | 2120.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 2117.45 | 2133.45 | 2120.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 2128.95 | 2132.55 | 2120.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:30:00 | 2124.00 | 2132.55 | 2120.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 2125.40 | 2130.66 | 2122.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:30:00 | 2121.00 | 2130.66 | 2122.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 12:15:00 | 2123.00 | 2129.81 | 2123.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 13:00:00 | 2123.00 | 2129.81 | 2123.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 13:15:00 | 2119.85 | 2127.81 | 2123.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 14:00:00 | 2119.85 | 2127.81 | 2123.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 2118.00 | 2124.28 | 2122.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 10:00:00 | 2113.30 | 2122.08 | 2121.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — SELL (started 2024-06-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 10:15:00 | 2112.70 | 2120.21 | 2120.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 11:15:00 | 2101.55 | 2116.48 | 2119.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 11:15:00 | 2107.60 | 2104.18 | 2110.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 11:15:00 | 2107.60 | 2104.18 | 2110.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 2107.60 | 2104.18 | 2110.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 2107.60 | 2104.18 | 2110.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 2071.55 | 2091.23 | 2101.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 2066.20 | 2091.23 | 2101.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 11:15:00 | 2111.00 | 2093.83 | 2100.49 | SL hit (close>static) qty=1.00 sl=2105.95 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 1999.85 | 1996.95 | 1996.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 2010.30 | 2001.05 | 1998.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 12:15:00 | 2009.80 | 2011.63 | 2005.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 12:15:00 | 2009.80 | 2011.63 | 2005.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 2009.80 | 2011.63 | 2005.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 2009.80 | 2011.63 | 2005.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 2022.55 | 2013.81 | 2007.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:45:00 | 2028.30 | 2017.18 | 2009.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 10:15:00 | 2028.00 | 2030.88 | 2023.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 15:15:00 | 2015.00 | 2020.76 | 2021.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2024-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 15:15:00 | 2015.00 | 2020.76 | 2021.29 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 09:15:00 | 2037.95 | 2024.20 | 2022.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 10:15:00 | 2041.10 | 2027.58 | 2024.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 14:15:00 | 2100.00 | 2109.14 | 2092.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:00:00 | 2100.00 | 2109.14 | 2092.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 2104.95 | 2108.30 | 2093.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 09:15:00 | 2111.00 | 2108.30 | 2093.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 2091.80 | 2105.00 | 2093.56 | SL hit (close<static) qty=1.00 sl=2092.50 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 09:15:00 | 2048.00 | 2088.92 | 2090.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 2039.60 | 2063.29 | 2071.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 13:15:00 | 2059.30 | 2055.41 | 2064.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-15 14:00:00 | 2059.30 | 2055.41 | 2064.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2069.65 | 2058.26 | 2065.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 14:45:00 | 2072.20 | 2058.26 | 2065.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 2065.75 | 2059.75 | 2065.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 2101.75 | 2059.75 | 2065.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 2109.00 | 2069.60 | 2069.12 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 2063.20 | 2078.13 | 2079.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 2048.90 | 2069.11 | 2074.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 2032.15 | 2029.13 | 2042.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:30:00 | 2035.30 | 2029.13 | 2042.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 2027.60 | 2025.40 | 2035.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 2027.60 | 2025.40 | 2035.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 13:15:00 | 2011.30 | 2023.78 | 2031.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 13:45:00 | 2009.65 | 2023.78 | 2031.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 2021.05 | 2023.24 | 2030.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 14:45:00 | 2036.10 | 2023.24 | 2030.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 2021.95 | 2019.74 | 2026.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 2021.00 | 2019.74 | 2026.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 2034.25 | 2022.65 | 2027.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 2034.25 | 2022.65 | 2027.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 2043.00 | 2026.72 | 2028.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 2048.30 | 2026.72 | 2028.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 14:15:00 | 2085.00 | 2038.37 | 2033.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 15:15:00 | 2146.55 | 2060.01 | 2043.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 09:15:00 | 2057.55 | 2059.52 | 2045.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 10:00:00 | 2057.55 | 2059.52 | 2045.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 2044.25 | 2056.46 | 2045.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 2044.25 | 2056.46 | 2045.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 2046.75 | 2054.52 | 2045.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:45:00 | 2045.00 | 2054.52 | 2045.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 2041.25 | 2051.87 | 2044.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:00:00 | 2041.25 | 2051.87 | 2044.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 2031.30 | 2047.75 | 2043.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:45:00 | 2035.00 | 2047.75 | 2043.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 2035.00 | 2045.20 | 2042.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:45:00 | 2026.95 | 2045.20 | 2042.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 09:15:00 | 2021.95 | 2039.86 | 2040.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 11:15:00 | 1992.95 | 2024.73 | 2033.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 13:15:00 | 2002.40 | 2002.08 | 2012.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-30 13:45:00 | 2000.05 | 2002.08 | 2012.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 2018.35 | 2005.34 | 2013.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 2018.35 | 2005.34 | 2013.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 2025.00 | 2009.27 | 2014.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:15:00 | 2022.55 | 2009.27 | 2014.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 2008.85 | 2010.58 | 2013.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:30:00 | 2003.05 | 2007.31 | 2011.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 2033.00 | 2013.92 | 2013.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 09:15:00 | 2033.00 | 2013.92 | 2013.45 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 2008.55 | 2013.58 | 2014.17 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 10:15:00 | 2018.70 | 2014.66 | 2014.56 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 2003.95 | 2012.52 | 2013.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 1998.75 | 2007.32 | 2010.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1970.95 | 1965.20 | 1981.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 09:45:00 | 1970.75 | 1965.20 | 1981.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1988.60 | 1969.88 | 1982.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 1988.60 | 1969.88 | 1982.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 1998.30 | 1975.56 | 1983.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 1985.55 | 1981.90 | 1985.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 1980.75 | 1981.90 | 1985.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 10:45:00 | 1985.40 | 1983.70 | 1985.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 1961.05 | 1925.53 | 1921.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1961.05 | 1925.53 | 1921.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1971.95 | 1945.89 | 1932.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 1987.75 | 1987.80 | 1965.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 1987.75 | 1987.80 | 1965.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 1952.00 | 1979.87 | 1965.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 1952.00 | 1979.87 | 1965.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 1949.00 | 1973.70 | 1963.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:30:00 | 1951.05 | 1973.70 | 1963.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 14:15:00 | 1935.80 | 1957.31 | 1958.53 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 1970.00 | 1958.87 | 1958.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 1981.00 | 1963.30 | 1960.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 10:15:00 | 2010.00 | 2019.99 | 2004.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 11:00:00 | 2010.00 | 2019.99 | 2004.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 2011.15 | 2018.22 | 2005.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 12:00:00 | 2020.20 | 2011.30 | 2008.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 13:00:00 | 2026.20 | 2035.13 | 2034.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 14:15:00 | 2031.05 | 2033.91 | 2034.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 2031.05 | 2033.91 | 2034.08 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 2039.45 | 2034.39 | 2034.23 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 2012.40 | 2031.05 | 2033.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 12:15:00 | 2009.60 | 2023.28 | 2028.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 2029.80 | 2023.71 | 2027.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 14:15:00 | 2029.80 | 2023.71 | 2027.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 2029.80 | 2023.71 | 2027.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 2029.80 | 2023.71 | 2027.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 2037.85 | 2026.54 | 2028.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 2035.15 | 2026.54 | 2028.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 2049.00 | 2031.03 | 2030.46 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 2018.50 | 2029.79 | 2030.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 2008.70 | 2021.95 | 2026.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 2031.75 | 2022.76 | 2026.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 2031.75 | 2022.76 | 2026.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 2031.75 | 2022.76 | 2026.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 2031.75 | 2022.76 | 2026.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 2042.10 | 2026.63 | 2027.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 2042.10 | 2026.63 | 2027.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 2066.35 | 2034.58 | 2031.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 13:15:00 | 2071.35 | 2047.04 | 2037.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 2076.50 | 2078.62 | 2066.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 11:45:00 | 2078.70 | 2078.62 | 2066.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 2074.90 | 2078.34 | 2072.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 2069.95 | 2078.34 | 2072.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 2068.60 | 2076.39 | 2072.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 2068.60 | 2076.39 | 2072.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 2073.55 | 2075.82 | 2072.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:45:00 | 2081.95 | 2077.43 | 2073.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 11:45:00 | 2080.35 | 2075.81 | 2073.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 2080.10 | 2076.28 | 2074.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 14:00:00 | 2080.75 | 2076.28 | 2074.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 2080.50 | 2077.12 | 2074.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 14:45:00 | 2075.25 | 2077.12 | 2074.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 2108.00 | 2083.38 | 2078.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 12:30:00 | 2113.00 | 2094.38 | 2085.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 14:45:00 | 2113.80 | 2100.35 | 2089.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:00:00 | 2110.00 | 2106.95 | 2095.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:30:00 | 2114.85 | 2108.51 | 2098.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 2085.75 | 2103.41 | 2097.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 2088.00 | 2103.41 | 2097.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 2091.10 | 2100.95 | 2097.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 2081.75 | 2100.95 | 2097.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 2074.65 | 2095.69 | 2095.10 | SL hit (close<static) qty=1.00 sl=2076.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 2060.85 | 2088.72 | 2091.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 2026.45 | 2071.34 | 2083.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 2113.15 | 2071.28 | 2078.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 09:15:00 | 2113.15 | 2071.28 | 2078.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2113.15 | 2071.28 | 2078.39 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 11:15:00 | 2112.35 | 2084.89 | 2083.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 12:15:00 | 2117.00 | 2091.31 | 2086.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 11:15:00 | 2078.40 | 2103.74 | 2097.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 2078.40 | 2103.74 | 2097.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 2078.40 | 2103.74 | 2097.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:00:00 | 2078.40 | 2103.74 | 2097.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 2076.00 | 2098.19 | 2095.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 12:45:00 | 2070.45 | 2098.19 | 2095.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-20 13:15:00 | 2072.00 | 2092.95 | 2093.44 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 2121.50 | 2098.82 | 2095.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 2130.80 | 2105.21 | 2098.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 2134.10 | 2143.27 | 2129.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 2134.10 | 2143.27 | 2129.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 2146.60 | 2143.94 | 2131.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 2148.25 | 2143.94 | 2131.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 2114.45 | 2138.04 | 2129.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 2114.45 | 2138.04 | 2129.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 2120.80 | 2134.59 | 2128.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 2132.45 | 2132.56 | 2128.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 11:15:00 | 2150.85 | 2165.27 | 2165.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 11:15:00 | 2150.85 | 2165.27 | 2165.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 2144.55 | 2157.89 | 2161.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 14:15:00 | 2151.05 | 2148.81 | 2155.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 2151.05 | 2148.81 | 2155.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 2150.45 | 2147.43 | 2153.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:00:00 | 2150.45 | 2147.43 | 2153.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 2161.15 | 2150.18 | 2154.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 2161.15 | 2150.18 | 2154.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 2142.70 | 2148.68 | 2153.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:30:00 | 2131.85 | 2145.79 | 2151.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:15:00 | 2025.26 | 2056.40 | 2070.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-18 15:15:00 | 1979.65 | 1978.56 | 1998.33 | SL hit (close>ema200) qty=0.50 sl=1978.56 alert=retest2 |

### Cycle 120 — BUY (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 14:15:00 | 1988.75 | 1966.46 | 1964.46 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1953.70 | 1967.58 | 1967.81 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 13:15:00 | 1982.40 | 1969.00 | 1968.32 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 09:15:00 | 1940.60 | 1965.62 | 1967.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1905.20 | 1937.70 | 1951.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 1932.80 | 1928.34 | 1940.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 15:00:00 | 1932.80 | 1928.34 | 1940.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1936.00 | 1929.23 | 1936.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 1936.00 | 1929.23 | 1936.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 13:15:00 | 1930.45 | 1929.48 | 1935.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:45:00 | 1934.20 | 1929.48 | 1935.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1927.95 | 1929.17 | 1934.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 14:30:00 | 1934.65 | 1929.17 | 1934.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1912.35 | 1925.61 | 1932.26 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 1948.65 | 1923.43 | 1922.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 1951.35 | 1933.75 | 1927.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1944.50 | 1956.62 | 1946.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1944.50 | 1956.62 | 1946.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1944.50 | 1956.62 | 1946.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1944.50 | 1956.62 | 1946.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1944.10 | 1954.12 | 1946.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1946.75 | 1954.12 | 1946.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1937.30 | 1950.75 | 1945.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1937.30 | 1950.75 | 1945.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1937.65 | 1948.13 | 1944.82 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 1930.00 | 1940.65 | 1941.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 12:15:00 | 1925.70 | 1933.82 | 1937.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 15:15:00 | 1932.00 | 1930.33 | 1935.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-11 09:15:00 | 1904.35 | 1930.33 | 1935.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1924.85 | 1927.01 | 1932.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 1929.00 | 1927.01 | 1932.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1919.05 | 1915.21 | 1922.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 1918.15 | 1915.21 | 1922.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1883.70 | 1874.66 | 1887.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 1887.60 | 1874.66 | 1887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1883.75 | 1876.48 | 1887.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 1889.10 | 1876.48 | 1887.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1868.05 | 1874.79 | 1885.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:15:00 | 1840.55 | 1878.40 | 1884.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 14:15:00 | 1882.30 | 1876.14 | 1880.22 | SL hit (close>ema400) qty=1.00 sl=1880.22 alert=retest1 |

### Cycle 126 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 1888.00 | 1882.98 | 1882.49 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 13:15:00 | 1864.10 | 1879.00 | 1880.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 14:15:00 | 1848.35 | 1872.87 | 1877.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 11:15:00 | 1848.00 | 1835.69 | 1847.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 11:15:00 | 1848.00 | 1835.69 | 1847.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1848.00 | 1835.69 | 1847.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 1848.00 | 1835.69 | 1847.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1858.00 | 1840.15 | 1848.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 1858.00 | 1840.15 | 1848.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1866.00 | 1845.32 | 1849.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1866.00 | 1845.32 | 1849.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 1869.85 | 1852.46 | 1852.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 1877.25 | 1852.46 | 1852.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1896.30 | 1861.23 | 1856.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1901.20 | 1869.22 | 1860.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1890.60 | 1899.42 | 1889.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1890.60 | 1899.42 | 1889.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1890.60 | 1899.42 | 1889.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 1883.30 | 1899.42 | 1889.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1894.55 | 1898.44 | 1890.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:30:00 | 1906.80 | 1900.53 | 1891.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 13:15:00 | 1950.00 | 1956.90 | 1957.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 1950.00 | 1956.90 | 1957.65 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 1990.90 | 1963.61 | 1960.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 2008.25 | 1992.08 | 1979.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 14:15:00 | 1986.40 | 1995.15 | 1986.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 14:15:00 | 1986.40 | 1995.15 | 1986.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 1986.40 | 1995.15 | 1986.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 15:00:00 | 1986.40 | 1995.15 | 1986.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1980.50 | 1992.22 | 1985.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 1976.90 | 1989.16 | 1984.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1989.30 | 1989.18 | 1985.33 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 13:15:00 | 1974.05 | 1981.78 | 1982.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 1968.05 | 1977.43 | 1980.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 1990.05 | 1977.32 | 1978.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 1990.05 | 1977.32 | 1978.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1990.05 | 1977.32 | 1978.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 1990.05 | 1977.32 | 1978.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 1998.35 | 1981.53 | 1980.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 2019.20 | 1992.77 | 1986.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 14:15:00 | 1997.95 | 1998.35 | 1991.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 14:15:00 | 1997.95 | 1998.35 | 1991.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 1997.95 | 1998.35 | 1991.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 15:00:00 | 1997.95 | 1998.35 | 1991.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 2001.20 | 1998.92 | 1992.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 09:45:00 | 2011.00 | 2000.94 | 1994.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 10:30:00 | 2012.15 | 2002.19 | 1995.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 15:15:00 | 2021.85 | 2004.64 | 1998.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 10:15:00 | 2012.55 | 2008.07 | 2001.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 2001.85 | 2007.49 | 2003.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:00:00 | 2001.85 | 2007.49 | 2003.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 2014.15 | 2008.82 | 2004.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 14:15:00 | 2018.75 | 2008.82 | 2004.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 15:00:00 | 2016.30 | 2010.32 | 2005.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 09:30:00 | 2016.95 | 2015.14 | 2008.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 12:00:00 | 2020.70 | 2016.36 | 2010.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 2013.50 | 2018.62 | 2013.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 2030.00 | 2020.03 | 2014.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:30:00 | 2032.90 | 2024.62 | 2017.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 14:00:00 | 2024.60 | 2027.13 | 2020.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:00:00 | 2026.00 | 2027.09 | 2022.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 2000.55 | 2021.78 | 2020.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 2000.55 | 2021.78 | 2020.12 | SL hit (close<static) qty=1.00 sl=2012.25 alert=retest2 |

### Cycle 133 — SELL (started 2024-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 12:15:00 | 2008.60 | 2017.55 | 2018.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 13:15:00 | 1992.95 | 2012.63 | 2016.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 2016.75 | 2009.58 | 2013.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 2016.75 | 2009.58 | 2013.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 2016.75 | 2009.58 | 2013.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 2016.75 | 2009.58 | 2013.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 2034.50 | 2014.56 | 2015.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:00:00 | 2034.50 | 2014.56 | 2015.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 11:15:00 | 2037.60 | 2019.17 | 2017.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-24 12:15:00 | 2044.25 | 2024.19 | 2019.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 2029.95 | 2035.31 | 2026.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 15:15:00 | 2029.95 | 2035.31 | 2026.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 2029.95 | 2035.31 | 2026.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:30:00 | 2024.80 | 2035.03 | 2027.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 2026.00 | 2033.22 | 2027.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 10:45:00 | 2025.00 | 2033.22 | 2027.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 2007.85 | 2028.15 | 2025.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 2007.85 | 2028.15 | 2025.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 2013.20 | 2025.16 | 2024.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 2022.50 | 2026.93 | 2025.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 15:15:00 | 2011.10 | 2022.61 | 2023.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 2011.10 | 2022.61 | 2023.61 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 2039.70 | 2026.03 | 2025.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 2055.95 | 2041.91 | 2037.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 2123.95 | 2124.27 | 2102.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 2123.95 | 2124.27 | 2102.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2113.40 | 2123.59 | 2109.32 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2067.40 | 2096.31 | 2099.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 2064.80 | 2075.51 | 2083.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 14:15:00 | 2028.00 | 2020.58 | 2039.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 15:00:00 | 2028.00 | 2020.58 | 2039.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 2031.05 | 2015.11 | 2025.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-10 15:00:00 | 2031.05 | 2015.11 | 2025.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 2025.00 | 2017.08 | 2025.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 2005.05 | 2017.08 | 2025.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 2069.70 | 1971.81 | 1959.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 2069.70 | 1971.81 | 1959.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 09:15:00 | 2091.10 | 2043.45 | 2005.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 15:15:00 | 2057.85 | 2061.42 | 2033.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-22 09:15:00 | 2044.55 | 2061.42 | 2033.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 2047.05 | 2058.55 | 2034.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 2074.00 | 2058.55 | 2034.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:00:00 | 2078.80 | 2063.87 | 2049.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 12:00:00 | 2070.70 | 2066.93 | 2053.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 13:30:00 | 2069.00 | 2068.00 | 2056.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2065.25 | 2068.91 | 2059.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2065.25 | 2068.91 | 2059.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2075.80 | 2070.29 | 2061.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 2081.00 | 2071.24 | 2062.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:15:00 | 2079.30 | 2071.24 | 2062.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:45:00 | 2079.70 | 2073.36 | 2064.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 2036.75 | 2063.20 | 2062.03 | SL hit (close<static) qty=1.00 sl=2059.25 alert=retest2 |

### Cycle 139 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 2040.00 | 2058.56 | 2060.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 2028.10 | 2045.63 | 2053.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 2028.55 | 2027.86 | 2040.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 2028.55 | 2027.86 | 2040.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 2030.05 | 2029.46 | 2038.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 2030.05 | 2029.46 | 2038.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 2025.60 | 2027.94 | 2036.41 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 11:15:00 | 2058.35 | 2043.37 | 2042.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 13:15:00 | 2071.00 | 2051.50 | 2046.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 2113.00 | 2128.04 | 2109.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 2113.00 | 2128.04 | 2109.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 2113.00 | 2128.04 | 2109.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:30:00 | 2112.50 | 2128.04 | 2109.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 2088.70 | 2120.18 | 2107.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 2088.70 | 2120.18 | 2107.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 2082.30 | 2112.60 | 2104.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:00:00 | 2082.30 | 2112.60 | 2104.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 2181.95 | 2214.15 | 2179.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:30:00 | 2180.00 | 2214.15 | 2179.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 2194.15 | 2210.15 | 2181.24 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 2124.40 | 2167.02 | 2169.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 09:15:00 | 2081.05 | 2135.26 | 2152.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 14:15:00 | 2073.00 | 2068.86 | 2093.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:30:00 | 2083.70 | 2068.86 | 2093.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2060.10 | 2053.72 | 2069.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 2062.00 | 2053.72 | 2069.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2058.50 | 2054.68 | 2068.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 2063.45 | 2054.68 | 2068.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 2034.00 | 2050.07 | 2060.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:45:00 | 2057.30 | 2050.07 | 2060.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2058.45 | 2039.56 | 2048.01 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 14:15:00 | 2060.95 | 2051.46 | 2051.33 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 15:15:00 | 2048.50 | 2050.87 | 2051.07 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 2056.00 | 2051.90 | 2051.52 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 10:15:00 | 2048.30 | 2051.18 | 2051.23 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 2053.00 | 2051.54 | 2051.39 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 13:15:00 | 2026.20 | 2047.22 | 2049.50 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 09:15:00 | 2099.95 | 2052.59 | 2050.88 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 14:15:00 | 2039.90 | 2049.89 | 2051.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 13:15:00 | 2023.60 | 2034.22 | 2041.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 2038.00 | 2031.97 | 2038.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 2038.00 | 2031.97 | 2038.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 2038.00 | 2031.97 | 2038.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:00:00 | 2022.00 | 2030.30 | 2036.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:30:00 | 2022.85 | 2028.83 | 2035.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 13:45:00 | 2023.35 | 2027.71 | 2034.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 14:30:00 | 2023.00 | 2026.85 | 2033.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 2011.35 | 2023.27 | 2030.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-20 11:15:00 | 2048.25 | 2030.28 | 2028.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 2048.25 | 2030.28 | 2028.93 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 2019.60 | 2027.97 | 2028.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 2018.60 | 2024.75 | 2026.75 | Break + close below crossover candle low |

### Cycle 152 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 2055.00 | 2028.46 | 2027.60 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 14:15:00 | 2021.70 | 2034.30 | 2035.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 15:15:00 | 1965.00 | 2020.44 | 2028.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 1897.95 | 1895.86 | 1910.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 09:15:00 | 1897.95 | 1895.86 | 1910.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1897.95 | 1895.86 | 1910.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-07 10:00:00 | 1885.20 | 1897.68 | 1904.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 1911.75 | 1899.82 | 1899.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-10 10:15:00 | 1911.75 | 1899.82 | 1899.07 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1890.10 | 1899.34 | 1899.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1866.80 | 1892.83 | 1896.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 1881.75 | 1878.92 | 1886.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 1881.75 | 1878.92 | 1886.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1886.15 | 1880.49 | 1886.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:30:00 | 1892.15 | 1880.49 | 1886.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 10:15:00 | 1901.65 | 1884.72 | 1887.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 10:45:00 | 1903.20 | 1884.72 | 1887.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1903.90 | 1888.56 | 1889.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 11:30:00 | 1899.50 | 1888.56 | 1889.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 12:15:00 | 1905.55 | 1891.96 | 1890.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1920.55 | 1903.69 | 1897.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 1902.80 | 1903.51 | 1897.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 11:00:00 | 1902.80 | 1903.51 | 1897.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1910.25 | 1904.88 | 1900.11 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 1877.60 | 1897.06 | 1897.62 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1900.00 | 1893.71 | 1893.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 11:15:00 | 1911.10 | 1900.38 | 1896.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 1904.85 | 1906.99 | 1901.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 14:15:00 | 1904.85 | 1906.99 | 1901.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 1904.85 | 1906.99 | 1901.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 1904.85 | 1906.99 | 1901.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 1905.00 | 1906.59 | 1901.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 09:15:00 | 1960.00 | 1906.59 | 1901.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 1929.45 | 1944.48 | 1945.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1929.45 | 1944.48 | 1945.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 13:15:00 | 1920.20 | 1939.62 | 1943.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 1943.95 | 1935.26 | 1939.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 1943.95 | 1935.26 | 1939.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 1943.95 | 1935.26 | 1939.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:30:00 | 1951.80 | 1935.26 | 1939.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 1925.95 | 1933.40 | 1938.57 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1950.00 | 1938.17 | 1938.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 1976.20 | 1948.43 | 1942.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 1977.70 | 1991.94 | 1978.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 10:15:00 | 1977.70 | 1991.94 | 1978.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1977.70 | 1991.94 | 1978.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:45:00 | 1977.70 | 1991.94 | 1978.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1979.00 | 1989.35 | 1978.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 1967.80 | 1989.35 | 1978.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 1951.15 | 1981.71 | 1976.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:00:00 | 1951.15 | 1981.71 | 1976.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1954.50 | 1976.27 | 1974.28 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 1951.80 | 1971.37 | 1972.24 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 11:15:00 | 1973.40 | 1972.10 | 1972.05 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 13:15:00 | 1963.30 | 1970.59 | 1971.38 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1975.00 | 1972.01 | 1971.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 10:15:00 | 1992.45 | 1976.49 | 1974.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 11:15:00 | 1973.50 | 1975.90 | 1973.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 1973.50 | 1975.90 | 1973.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1973.50 | 1975.90 | 1973.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:45:00 | 1972.40 | 1975.90 | 1973.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1981.30 | 1976.98 | 1974.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 13:00:00 | 1991.30 | 1982.30 | 1978.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1907.60 | 1970.85 | 1975.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1907.60 | 1970.85 | 1975.09 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1997.35 | 1968.38 | 1966.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 2014.35 | 1977.57 | 1970.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-23 15:15:00 | 2232.00 | 2235.77 | 2214.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 09:15:00 | 2221.00 | 2235.77 | 2214.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2209.00 | 2230.42 | 2214.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 2209.00 | 2230.42 | 2214.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 2217.00 | 2227.73 | 2214.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:45:00 | 2208.00 | 2227.73 | 2214.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 2197.70 | 2221.73 | 2212.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 12:00:00 | 2197.70 | 2221.73 | 2212.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 12:15:00 | 2180.60 | 2213.50 | 2209.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:00:00 | 2180.60 | 2213.50 | 2209.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 2177.10 | 2206.22 | 2206.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 2136.40 | 2182.34 | 2194.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2145.40 | 2141.85 | 2163.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 14:15:00 | 2156.00 | 2144.15 | 2156.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 2156.00 | 2144.15 | 2156.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 2156.00 | 2144.15 | 2156.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 2161.50 | 2147.62 | 2156.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 2144.00 | 2147.62 | 2156.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2145.50 | 2147.20 | 2155.77 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 14:15:00 | 2173.00 | 2159.72 | 2159.00 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 2145.10 | 2157.30 | 2158.75 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 15:15:00 | 2165.00 | 2158.52 | 2158.17 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 13:15:00 | 2137.90 | 2156.01 | 2157.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 2132.20 | 2145.60 | 2151.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 15:15:00 | 2150.00 | 2137.44 | 2143.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 15:15:00 | 2150.00 | 2137.44 | 2143.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 2150.00 | 2137.44 | 2143.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:00:00 | 2131.00 | 2136.15 | 2142.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 2163.00 | 2146.64 | 2145.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 2163.00 | 2146.64 | 2145.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 15:15:00 | 2195.00 | 2161.84 | 2153.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 2165.00 | 2166.92 | 2157.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 10:15:00 | 2165.00 | 2166.92 | 2157.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 2165.00 | 2166.92 | 2157.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:00:00 | 2165.00 | 2166.92 | 2157.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 2163.50 | 2166.23 | 2157.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:45:00 | 2164.30 | 2166.23 | 2157.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 2156.60 | 2164.31 | 2157.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 2156.60 | 2164.31 | 2157.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 2161.80 | 2163.80 | 2158.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 2143.80 | 2163.80 | 2158.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 2177.00 | 2166.44 | 2159.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 2155.40 | 2166.44 | 2159.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2159.00 | 2164.95 | 2159.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 2103.90 | 2164.95 | 2159.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 2111.10 | 2154.18 | 2155.35 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 10:15:00 | 2170.00 | 2157.35 | 2156.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 11:15:00 | 2174.80 | 2160.84 | 2158.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-12 09:15:00 | 2144.90 | 2162.30 | 2160.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 2144.90 | 2162.30 | 2160.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2144.90 | 2162.30 | 2160.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 2137.20 | 2162.30 | 2160.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 10:15:00 | 2138.40 | 2157.52 | 2158.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-12 12:15:00 | 2126.40 | 2147.71 | 2153.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 2074.90 | 2074.59 | 2102.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 2074.50 | 2073.13 | 2087.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 2074.50 | 2073.13 | 2087.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 11:00:00 | 2058.00 | 2070.10 | 2085.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:15:00 | 2054.00 | 2075.23 | 2083.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 09:45:00 | 2060.50 | 2067.81 | 2078.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 2060.00 | 2050.67 | 2060.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 2053.00 | 2051.13 | 2060.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 2049.10 | 2050.93 | 2059.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-19 15:15:00 | 2045.00 | 2051.14 | 2058.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 2047.30 | 2050.19 | 2056.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 2048.40 | 2050.31 | 2056.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 2052.50 | 2039.57 | 2047.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:45:00 | 2027.50 | 2037.25 | 2044.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 2013.70 | 2033.83 | 2040.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 2039.50 | 2022.62 | 2021.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 2039.50 | 2022.62 | 2021.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 2045.70 | 2027.24 | 2023.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2021.10 | 2034.16 | 2029.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2021.10 | 2034.16 | 2029.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2017.70 | 2030.87 | 2028.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 2017.40 | 2030.87 | 2028.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 2011.00 | 2024.52 | 2025.75 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 2043.10 | 2029.22 | 2027.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 11:15:00 | 2051.00 | 2033.97 | 2029.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 2026.80 | 2043.20 | 2037.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 2030.00 | 2043.20 | 2037.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 2022.80 | 2039.12 | 2036.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 2026.30 | 2039.12 | 2036.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 2013.10 | 2031.02 | 2032.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 2002.40 | 2021.99 | 2028.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 2007.00 | 1994.00 | 2006.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 2007.00 | 1994.00 | 2006.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 2001.90 | 1995.58 | 2005.65 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 2025.00 | 2011.63 | 2009.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 2036.20 | 2020.29 | 2014.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 2030.00 | 2030.52 | 2025.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 14:00:00 | 2030.00 | 2030.52 | 2025.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 2016.00 | 2027.10 | 2024.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:30:00 | 2032.30 | 2030.62 | 2026.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 15:15:00 | 2055.10 | 2062.89 | 2063.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 2055.10 | 2062.89 | 2063.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 2037.70 | 2057.74 | 2060.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 2053.80 | 2043.01 | 2050.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 2053.80 | 2043.01 | 2050.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 2040.40 | 2042.49 | 2049.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:30:00 | 2034.90 | 2041.15 | 2048.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1933.15 | 1965.95 | 1991.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1963.00 | 1960.15 | 1981.72 | SL hit (close>ema200) qty=0.50 sl=1960.15 alert=retest2 |

### Cycle 182 — BUY (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 11:15:00 | 1955.10 | 1940.42 | 1939.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 12:15:00 | 1963.40 | 1945.02 | 1941.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1954.40 | 1963.11 | 1955.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1954.40 | 1963.11 | 1955.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1960.00 | 1962.49 | 1956.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 1976.00 | 1962.49 | 1956.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 12:15:00 | 1949.90 | 1965.57 | 1960.32 | SL hit (close<static) qty=1.00 sl=1950.10 alert=retest2 |

### Cycle 183 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 1947.20 | 1958.16 | 1958.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 1935.40 | 1948.55 | 1953.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1960.00 | 1947.28 | 1950.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1960.00 | 1947.28 | 1950.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 184 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1977.10 | 1953.24 | 1952.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 12:15:00 | 1981.20 | 1963.12 | 1957.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 09:15:00 | 1971.20 | 1972.55 | 1964.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 10:00:00 | 1971.20 | 1972.55 | 1964.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1965.00 | 1970.15 | 1964.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 1965.00 | 1970.15 | 1964.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1969.30 | 1969.98 | 1965.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1963.30 | 1969.98 | 1965.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1979.20 | 1985.70 | 1979.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1981.90 | 1985.70 | 1979.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1980.00 | 1984.56 | 1979.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1979.30 | 1984.56 | 1979.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1978.60 | 1983.37 | 1979.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:00:00 | 1978.60 | 1983.37 | 1979.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 1975.00 | 1981.70 | 1978.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:00:00 | 1975.00 | 1981.70 | 1978.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 1975.10 | 1980.38 | 1978.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:45:00 | 1976.00 | 1980.38 | 1978.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1990.00 | 1982.32 | 1979.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1972.70 | 1982.32 | 1979.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1964.60 | 1978.78 | 1978.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 1964.60 | 1978.78 | 1978.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 1966.90 | 1976.40 | 1977.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 1949.90 | 1969.12 | 1973.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1970.10 | 1968.49 | 1972.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 14:45:00 | 1972.90 | 1968.49 | 1972.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1942.50 | 1963.62 | 1969.71 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 1985.20 | 1961.10 | 1960.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 2000.70 | 1979.53 | 1971.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 11:15:00 | 2009.30 | 2016.63 | 2006.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 12:00:00 | 2009.30 | 2016.63 | 2006.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1994.30 | 2013.25 | 2006.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 1994.30 | 2013.25 | 2006.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2002.50 | 2011.10 | 2006.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 15:15:00 | 2009.50 | 2011.10 | 2006.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:45:00 | 2010.60 | 2010.50 | 2006.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:45:00 | 2019.00 | 2010.43 | 2007.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 2017.90 | 2024.20 | 2024.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 2017.90 | 2024.20 | 2024.92 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 2034.40 | 2025.84 | 2024.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 15:15:00 | 2037.90 | 2028.25 | 2026.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2021.70 | 2026.94 | 2025.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 2021.70 | 2026.94 | 2025.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 2014.10 | 2024.37 | 2024.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1997.80 | 2019.06 | 2022.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 1959.00 | 1956.83 | 1970.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 1959.00 | 1956.83 | 1970.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1939.00 | 1933.50 | 1942.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 1940.20 | 1933.50 | 1942.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1938.60 | 1934.52 | 1942.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1940.70 | 1934.52 | 1942.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1935.00 | 1934.61 | 1941.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1932.00 | 1934.61 | 1941.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1923.40 | 1932.37 | 1939.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:45:00 | 1919.20 | 1925.46 | 1931.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:45:00 | 1920.10 | 1925.42 | 1929.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:45:00 | 1919.00 | 1925.75 | 1929.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1935.30 | 1930.48 | 1930.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 1935.30 | 1930.48 | 1930.42 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 1920.40 | 1929.32 | 1929.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 09:15:00 | 1910.00 | 1920.14 | 1924.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1920.00 | 1915.08 | 1919.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1920.00 | 1915.08 | 1919.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1920.00 | 1916.07 | 1919.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1912.50 | 1916.07 | 1919.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1921.00 | 1917.05 | 1919.79 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1925.20 | 1922.17 | 1921.80 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1914.30 | 1920.45 | 1921.11 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 1924.00 | 1921.37 | 1921.26 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 1914.30 | 1919.95 | 1920.62 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 1927.60 | 1922.15 | 1921.51 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1893.70 | 1917.35 | 1919.74 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 1926.50 | 1916.67 | 1915.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 14:15:00 | 1929.10 | 1918.23 | 1916.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1915.00 | 1918.04 | 1916.83 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 1904.10 | 1915.26 | 1915.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 1890.00 | 1910.20 | 1913.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 1857.70 | 1855.62 | 1867.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 12:00:00 | 1857.70 | 1855.62 | 1867.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1834.60 | 1850.14 | 1860.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 1822.00 | 1845.95 | 1851.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:45:00 | 1821.10 | 1812.52 | 1827.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1847.60 | 1833.07 | 1832.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1847.60 | 1833.07 | 1832.29 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 1829.90 | 1832.32 | 1832.37 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 1847.40 | 1834.70 | 1833.40 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 13:15:00 | 1828.90 | 1833.72 | 1834.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 1825.30 | 1832.03 | 1833.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1831.80 | 1831.02 | 1832.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 1814.20 | 1824.37 | 1828.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1809.20 | 1813.84 | 1821.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:00:00 | 1810.30 | 1809.12 | 1815.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 1832.90 | 1806.18 | 1803.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1832.90 | 1806.18 | 1803.21 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 14:15:00 | 1799.60 | 1810.10 | 1811.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 1795.20 | 1803.64 | 1807.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1800.50 | 1785.93 | 1790.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:45:00 | 1800.90 | 1785.93 | 1790.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1797.20 | 1788.18 | 1791.13 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 09:15:00 | 1807.60 | 1795.15 | 1793.77 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1786.50 | 1792.59 | 1792.80 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 1797.30 | 1793.65 | 1793.19 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 09:15:00 | 1775.80 | 1790.45 | 1791.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 1763.00 | 1784.96 | 1789.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1783.50 | 1778.60 | 1784.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1783.50 | 1778.60 | 1784.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1783.70 | 1779.62 | 1784.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 1765.30 | 1779.62 | 1784.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1770.50 | 1777.80 | 1782.85 | EMA400 retest candle locked (from downside) |

### Cycle 210 — BUY (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 12:15:00 | 1795.00 | 1783.30 | 1782.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 1798.00 | 1786.24 | 1783.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1794.40 | 1797.68 | 1793.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 1815.80 | 1796.87 | 1794.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 1788.80 | 1797.27 | 1797.02 | SL hit (close<static) qty=1.00 sl=1790.70 alert=retest2 |

### Cycle 211 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1789.10 | 1795.64 | 1796.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1777.90 | 1787.19 | 1790.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1760.00 | 1756.37 | 1766.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 1760.00 | 1756.37 | 1766.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1777.30 | 1761.90 | 1767.13 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1777.40 | 1770.81 | 1770.40 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 1759.60 | 1769.02 | 1770.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1756.00 | 1762.54 | 1766.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1771.20 | 1762.40 | 1763.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1771.20 | 1762.40 | 1763.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1770.00 | 1763.92 | 1764.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1770.30 | 1763.92 | 1764.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1763.90 | 1762.81 | 1763.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:45:00 | 1766.20 | 1762.81 | 1763.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1754.00 | 1761.05 | 1762.81 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1779.80 | 1764.35 | 1763.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 1787.90 | 1773.93 | 1769.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1773.70 | 1780.09 | 1774.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 1777.00 | 1780.09 | 1774.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1781.00 | 1780.27 | 1774.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 15:15:00 | 1792.00 | 1780.27 | 1774.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 1785.80 | 1782.85 | 1777.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 1784.50 | 1822.58 | 1826.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 215 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 1784.50 | 1822.58 | 1826.55 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 1810.00 | 1805.98 | 1805.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 1815.60 | 1807.90 | 1806.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 1805.20 | 1809.01 | 1807.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:00:00 | 1805.20 | 1809.01 | 1807.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1805.70 | 1808.34 | 1807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:30:00 | 1799.40 | 1808.34 | 1807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1805.80 | 1809.45 | 1808.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:30:00 | 1802.80 | 1809.45 | 1808.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1806.50 | 1808.86 | 1808.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:45:00 | 1800.90 | 1808.86 | 1808.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 12:15:00 | 1803.00 | 1806.78 | 1807.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 13:15:00 | 1801.20 | 1805.67 | 1806.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1808.50 | 1805.54 | 1806.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1789.80 | 1805.54 | 1806.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1792.00 | 1785.44 | 1785.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1792.00 | 1785.44 | 1785.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 1793.00 | 1786.95 | 1786.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1795.10 | 1803.62 | 1798.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1795.10 | 1803.62 | 1798.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1811.90 | 1805.28 | 1799.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:30:00 | 1810.00 | 1805.28 | 1799.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1806.10 | 1805.44 | 1800.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:45:00 | 1811.40 | 1807.80 | 1803.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1797.70 | 1805.77 | 1803.69 | SL hit (close<static) qty=1.00 sl=1798.20 alert=retest2 |

### Cycle 219 — SELL (started 2025-11-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 14:15:00 | 1789.00 | 1800.57 | 1801.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 15:15:00 | 1784.00 | 1797.25 | 1799.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 1699.10 | 1698.60 | 1711.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 14:30:00 | 1698.30 | 1698.60 | 1711.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1695.30 | 1698.38 | 1704.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 1686.60 | 1696.60 | 1700.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 1689.10 | 1691.68 | 1696.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 14:00:00 | 1689.70 | 1691.28 | 1695.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:45:00 | 1678.10 | 1688.24 | 1693.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1690.80 | 1680.27 | 1685.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1695.60 | 1680.27 | 1685.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1690.00 | 1682.21 | 1685.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:45:00 | 1691.00 | 1682.21 | 1685.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 1701.90 | 1687.72 | 1687.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1701.90 | 1687.72 | 1687.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 13:15:00 | 1715.40 | 1693.25 | 1690.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1700.30 | 1702.43 | 1695.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 09:30:00 | 1699.60 | 1702.43 | 1695.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1700.00 | 1701.94 | 1696.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 1700.00 | 1701.94 | 1696.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1705.80 | 1704.21 | 1699.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:30:00 | 1705.30 | 1704.21 | 1699.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1730.90 | 1709.85 | 1702.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:15:00 | 1735.10 | 1709.85 | 1702.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 1689.20 | 1701.92 | 1702.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1689.20 | 1701.92 | 1702.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1672.10 | 1695.96 | 1699.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 1663.00 | 1657.35 | 1669.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 14:00:00 | 1663.00 | 1657.35 | 1669.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1664.20 | 1658.81 | 1667.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:15:00 | 1665.50 | 1658.81 | 1667.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1663.00 | 1659.65 | 1666.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1658.50 | 1659.74 | 1665.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1658.00 | 1659.33 | 1664.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:00:00 | 1657.70 | 1659.33 | 1664.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1653.30 | 1660.84 | 1664.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1647.20 | 1658.11 | 1662.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1637.80 | 1650.13 | 1656.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 1637.00 | 1648.13 | 1654.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1638.40 | 1647.78 | 1651.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:45:00 | 1637.00 | 1645.48 | 1650.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1575.57 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1575.10 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-19 10:15:00 | 1574.82 | 1588.17 | 1601.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1602.40 | 1586.15 | 1595.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1602.40 | 1586.15 | 1595.68 | SL hit (close>ema200) qty=0.50 sl=1586.15 alert=retest2 |

### Cycle 222 — BUY (started 2025-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 12:15:00 | 1611.70 | 1601.45 | 1600.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 1617.70 | 1606.82 | 1603.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1615.10 | 1621.19 | 1616.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1612.40 | 1621.19 | 1616.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1607.90 | 1618.53 | 1616.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1607.90 | 1618.53 | 1616.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1608.00 | 1616.43 | 1615.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:00:00 | 1620.00 | 1617.14 | 1615.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1611.00 | 1616.49 | 1616.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1611.00 | 1616.49 | 1616.52 | EMA200 below EMA400 |

### Cycle 224 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 1620.20 | 1616.67 | 1616.41 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 1613.00 | 1615.94 | 1616.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 1605.40 | 1611.60 | 1613.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1616.60 | 1611.43 | 1613.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1600.00 | 1611.39 | 1612.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 1620.00 | 1614.30 | 1613.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1620.00 | 1614.30 | 1613.94 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 1607.00 | 1613.23 | 1614.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 09:15:00 | 1596.80 | 1609.39 | 1612.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1604.10 | 1599.34 | 1604.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:45:00 | 1607.90 | 1600.07 | 1604.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1603.00 | 1600.66 | 1604.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 1593.80 | 1599.96 | 1603.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1514.11 | 1541.43 | 1554.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1526.80 | 1523.73 | 1536.75 | SL hit (close>ema200) qty=0.50 sl=1523.73 alert=retest2 |

### Cycle 228 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1444.90 | 1436.98 | 1436.20 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 14:15:00 | 1426.50 | 1436.29 | 1436.43 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 1476.20 | 1444.05 | 1439.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1492.00 | 1466.04 | 1459.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1492.60 | 1504.53 | 1493.90 | EMA400 retest candle locked (from upside) |

### Cycle 231 — SELL (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 15:15:00 | 1486.00 | 1489.84 | 1490.06 | EMA200 below EMA400 |

### Cycle 232 — BUY (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 09:15:00 | 1511.90 | 1494.25 | 1492.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 11:15:00 | 1517.30 | 1501.76 | 1496.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1511.00 | 1515.94 | 1506.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 1506.10 | 1515.94 | 1506.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1598.00 | 1615.12 | 1602.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1574.80 | 1615.12 | 1602.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1640.60 | 1620.21 | 1605.71 | EMA400 retest candle locked (from upside) |

### Cycle 233 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 1600.00 | 1609.62 | 1610.31 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 1617.00 | 1608.17 | 1607.83 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1602.00 | 1608.21 | 1608.70 | EMA200 below EMA400 |

### Cycle 236 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 1613.60 | 1609.31 | 1608.72 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 1604.70 | 1608.55 | 1608.69 | EMA200 below EMA400 |

### Cycle 238 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1616.50 | 1610.14 | 1609.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 10:15:00 | 1623.60 | 1612.83 | 1610.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1605.30 | 1619.35 | 1615.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1603.70 | 1619.35 | 1615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 1600.00 | 1615.48 | 1614.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 1600.00 | 1615.48 | 1614.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 239 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 1604.10 | 1613.21 | 1613.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 14:15:00 | 1596.60 | 1606.48 | 1610.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 1594.50 | 1591.87 | 1597.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 1595.20 | 1591.87 | 1597.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 1597.00 | 1591.03 | 1594.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 1590.60 | 1591.03 | 1594.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1587.50 | 1590.32 | 1594.07 | EMA400 retest candle locked (from downside) |

### Cycle 240 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 1605.20 | 1596.54 | 1595.78 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1582.30 | 1593.45 | 1594.49 | EMA200 below EMA400 |

### Cycle 242 — BUY (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 12:15:00 | 1597.30 | 1592.61 | 1592.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-04 13:15:00 | 1605.00 | 1595.09 | 1593.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1704.40 | 1705.49 | 1667.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1696.70 | 1712.29 | 1691.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1696.70 | 1712.29 | 1691.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 1682.00 | 1712.29 | 1691.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 1697.30 | 1708.43 | 1695.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:45:00 | 1695.90 | 1708.43 | 1695.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 1698.00 | 1706.35 | 1696.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:30:00 | 1692.40 | 1706.35 | 1696.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1694.20 | 1703.92 | 1695.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1686.10 | 1703.92 | 1695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1690.20 | 1701.17 | 1695.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 1682.40 | 1701.17 | 1695.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1687.50 | 1698.44 | 1694.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 1687.50 | 1698.44 | 1694.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 243 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 1665.10 | 1689.53 | 1691.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1655.00 | 1679.08 | 1685.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1671.10 | 1657.71 | 1670.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1671.10 | 1657.71 | 1670.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 1659.50 | 1658.07 | 1669.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 1679.20 | 1658.07 | 1669.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1655.00 | 1637.26 | 1648.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 1593.90 | 1637.26 | 1648.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 1621.10 | 1625.81 | 1631.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1540.04 | 1573.38 | 1587.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 1560.70 | 1556.88 | 1568.00 | SL hit (close>ema200) qty=0.50 sl=1556.88 alert=retest2 |

### Cycle 244 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1620.40 | 1581.64 | 1576.67 | EMA200 above EMA400 |

### Cycle 245 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 1560.30 | 1588.06 | 1589.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1575.03 | 1582.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1585.30 | 1559.59 | 1568.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 1589.80 | 1559.59 | 1568.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1572.60 | 1562.19 | 1568.89 | EMA400 retest candle locked (from downside) |

### Cycle 246 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1579.10 | 1572.31 | 1572.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 1590.10 | 1575.87 | 1573.79 | Break + close above crossover candle high |

### Cycle 247 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1493.30 | 1559.36 | 1566.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 11:15:00 | 1481.80 | 1532.73 | 1552.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 1480.70 | 1477.15 | 1502.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 15:00:00 | 1480.70 | 1477.15 | 1502.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1506.20 | 1482.02 | 1490.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:30:00 | 1510.00 | 1482.02 | 1490.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1516.00 | 1488.82 | 1493.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 1516.00 | 1488.82 | 1493.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 1486.00 | 1492.93 | 1494.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 1479.00 | 1492.93 | 1494.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 10:15:00 | 1482.00 | 1467.47 | 1466.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 248 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1482.00 | 1467.47 | 1466.33 | EMA200 above EMA400 |

### Cycle 249 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 1460.90 | 1465.91 | 1466.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 1452.20 | 1463.16 | 1465.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 15:15:00 | 1459.50 | 1459.28 | 1462.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:15:00 | 1460.50 | 1459.28 | 1462.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 250 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 1490.70 | 1465.56 | 1464.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 11:15:00 | 1509.10 | 1475.62 | 1469.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1515.00 | 1516.77 | 1500.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1510.20 | 1516.77 | 1500.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1506.50 | 1515.60 | 1503.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1507.00 | 1515.60 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1502.00 | 1512.88 | 1502.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 1501.50 | 1512.88 | 1502.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1504.80 | 1511.26 | 1503.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1510.00 | 1505.91 | 1502.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 10:30:00 | 1509.30 | 1505.54 | 1502.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1507.00 | 1505.51 | 1503.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:00:00 | 1507.30 | 1505.87 | 1503.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1501.60 | 1504.89 | 1503.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:00:00 | 1501.60 | 1504.89 | 1503.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1500.00 | 1503.91 | 1503.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1488.00 | 1503.91 | 1503.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1493.40 | 1501.81 | 1502.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 251 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1493.40 | 1501.81 | 1502.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1482.50 | 1493.12 | 1497.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1487.80 | 1480.14 | 1485.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1489.70 | 1480.14 | 1485.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1490.90 | 1482.29 | 1486.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 1483.00 | 1482.29 | 1486.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:00:00 | 1487.20 | 1483.98 | 1486.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 1486.90 | 1482.98 | 1485.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 1483.60 | 1483.80 | 1485.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1483.20 | 1483.68 | 1485.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 1474.00 | 1482.39 | 1484.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:45:00 | 1472.40 | 1480.55 | 1482.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1408.85 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1412.84 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1412.56 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1409.42 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1400.30 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-06 09:15:00 | 1398.78 | 1443.84 | 1453.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1441.50 | 1424.24 | 1435.47 | SL hit (close>ema200) qty=0.50 sl=1424.24 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 1839.90 | 2024-04-19 09:15:00 | 1835.75 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-04-23 14:30:00 | 1854.45 | 2024-04-24 15:15:00 | 2039.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1981.90 | 2024-05-15 13:15:00 | 1882.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1981.90 | 2024-05-16 14:15:00 | 1899.40 | STOP_HIT | 0.50 | 4.16% |
| SELL | retest2 | 2024-05-09 09:15:00 | 1961.50 | 2024-05-17 14:15:00 | 1921.60 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2024-05-23 10:30:00 | 1870.50 | 2024-05-29 12:15:00 | 1877.95 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-05-24 09:15:00 | 1870.75 | 2024-05-29 12:15:00 | 1877.95 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-05-24 11:00:00 | 1872.15 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-05-24 15:00:00 | 1874.05 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-05-27 14:15:00 | 1860.25 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-27 15:00:00 | 1863.35 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-05-28 11:30:00 | 1860.30 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-28 12:30:00 | 1862.40 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-05-29 09:15:00 | 1860.15 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-05-29 11:45:00 | 1864.15 | 2024-05-29 14:15:00 | 1877.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-05-31 12:15:00 | 1848.90 | 2024-06-03 09:15:00 | 1865.95 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-06-19 10:15:00 | 2066.20 | 2024-06-19 11:15:00 | 2111.00 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2066.25 | 2024-06-26 15:15:00 | 1962.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 15:00:00 | 2066.25 | 2024-06-27 12:15:00 | 1985.10 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2024-06-25 09:15:00 | 2017.30 | 2024-07-01 11:15:00 | 1999.85 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2024-07-02 14:45:00 | 2028.30 | 2024-07-04 15:15:00 | 2015.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-04 10:15:00 | 2028.00 | 2024-07-04 15:15:00 | 2015.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-07-10 09:15:00 | 2111.00 | 2024-07-10 09:15:00 | 2091.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-07-10 15:15:00 | 2110.00 | 2024-07-11 09:15:00 | 2048.00 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-07-31 12:30:00 | 2003.05 | 2024-08-01 09:15:00 | 2033.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2024-08-06 14:30:00 | 1985.55 | 2024-08-16 11:15:00 | 1961.05 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2024-08-06 15:00:00 | 1980.75 | 2024-08-16 11:15:00 | 1961.05 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-08-07 10:45:00 | 1985.40 | 2024-08-16 11:15:00 | 1961.05 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2024-08-27 12:00:00 | 2020.20 | 2024-09-02 14:15:00 | 2031.05 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-09-02 13:00:00 | 2026.20 | 2024-09-02 14:15:00 | 2031.05 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-09-12 14:45:00 | 2081.95 | 2024-09-18 09:15:00 | 2074.65 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-09-13 11:45:00 | 2080.35 | 2024-09-18 09:15:00 | 2074.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-09-13 13:30:00 | 2080.10 | 2024-09-18 09:15:00 | 2074.65 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-09-13 14:00:00 | 2080.75 | 2024-09-18 09:15:00 | 2074.65 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-09-16 12:30:00 | 2113.00 | 2024-09-18 10:15:00 | 2060.85 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-09-16 14:45:00 | 2113.80 | 2024-09-18 10:15:00 | 2060.85 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-09-17 11:00:00 | 2110.00 | 2024-09-18 10:15:00 | 2060.85 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-09-17 12:30:00 | 2114.85 | 2024-09-18 10:15:00 | 2060.85 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2024-09-25 15:00:00 | 2132.45 | 2024-10-01 11:15:00 | 2150.85 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-10-04 12:30:00 | 2131.85 | 2024-10-16 10:15:00 | 2025.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-04 12:30:00 | 2131.85 | 2024-10-18 15:15:00 | 1979.65 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest1 | 2024-11-11 09:15:00 | 1904.35 | 2024-11-18 14:15:00 | 1882.30 | STOP_HIT | 1.00 | 1.16% |
| SELL | retest2 | 2024-11-18 09:15:00 | 1840.55 | 2024-11-19 09:15:00 | 1890.00 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-11-27 11:30:00 | 1906.80 | 2024-12-09 13:15:00 | 1950.00 | STOP_HIT | 1.00 | 2.27% |
| BUY | retest2 | 2024-12-17 09:45:00 | 2011.00 | 2024-12-23 10:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-12-17 10:30:00 | 2012.15 | 2024-12-23 10:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-12-17 15:15:00 | 2021.85 | 2024-12-23 10:15:00 | 2000.55 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-12-18 10:15:00 | 2012.55 | 2024-12-23 10:15:00 | 2000.55 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-12-18 14:15:00 | 2018.75 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-18 15:00:00 | 2016.30 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-12-19 09:30:00 | 2016.95 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-12-19 12:00:00 | 2020.70 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-12-20 09:30:00 | 2030.00 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-12-20 10:30:00 | 2032.90 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-12-20 14:00:00 | 2024.60 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-12-23 10:00:00 | 2026.00 | 2024-12-23 12:15:00 | 2008.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-12-26 13:45:00 | 2022.50 | 2024-12-26 15:15:00 | 2011.10 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-01-13 09:15:00 | 2005.05 | 2025-01-20 10:15:00 | 2069.70 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2025-01-22 10:15:00 | 2074.00 | 2025-01-27 09:15:00 | 2036.75 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-01-23 10:00:00 | 2078.80 | 2025-01-27 09:15:00 | 2036.75 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-01-23 12:00:00 | 2070.70 | 2025-01-27 09:15:00 | 2036.75 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-01-23 13:30:00 | 2069.00 | 2025-01-27 10:15:00 | 2040.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-01-24 11:30:00 | 2081.00 | 2025-01-27 10:15:00 | 2040.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-01-24 12:15:00 | 2079.30 | 2025-01-27 10:15:00 | 2040.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-01-24 12:45:00 | 2079.70 | 2025-01-27 10:15:00 | 2040.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-02-18 12:00:00 | 2022.00 | 2025-02-20 11:15:00 | 2048.25 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-02-18 12:30:00 | 2022.85 | 2025-02-20 11:15:00 | 2048.25 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-02-18 13:45:00 | 2023.35 | 2025-02-20 11:15:00 | 2048.25 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-02-18 14:30:00 | 2023.00 | 2025-02-20 11:15:00 | 2048.25 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-03-07 10:00:00 | 1885.20 | 2025-03-10 10:15:00 | 1911.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-03-20 09:15:00 | 1960.00 | 2025-03-25 12:15:00 | 1929.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-04-04 13:00:00 | 1991.30 | 2025-04-07 09:15:00 | 1907.60 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-05-07 10:00:00 | 2131.00 | 2025-05-07 13:15:00 | 2163.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-15 11:00:00 | 2058.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-05-15 15:15:00 | 2054.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-05-16 09:45:00 | 2060.50 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2025-05-19 11:45:00 | 2060.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-05-19 14:15:00 | 2049.10 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-05-19 15:15:00 | 2045.00 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-05-20 10:15:00 | 2047.30 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-05-20 11:15:00 | 2048.40 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-05-21 12:45:00 | 2027.50 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-05-22 09:15:00 | 2013.70 | 2025-05-26 10:15:00 | 2039.50 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-06 10:30:00 | 2032.30 | 2025-06-11 15:15:00 | 2055.10 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2034.90 | 2025-06-19 12:15:00 | 1933.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-16 09:30:00 | 2034.90 | 2025-06-19 15:15:00 | 1963.00 | STOP_HIT | 0.50 | 3.53% |
| BUY | retest2 | 2025-06-30 09:15:00 | 1976.00 | 2025-06-30 12:15:00 | 1949.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-01 09:15:00 | 1976.80 | 2025-07-01 12:15:00 | 1947.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-07-18 15:15:00 | 2009.50 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-07-21 09:45:00 | 2010.60 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-07-21 12:45:00 | 2019.00 | 2025-07-23 14:15:00 | 2017.90 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-08-06 11:45:00 | 1919.20 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-06 14:45:00 | 1920.10 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-08-07 09:45:00 | 1919.00 | 2025-08-07 14:15:00 | 1935.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-01 09:30:00 | 1822.00 | 2025-09-03 09:15:00 | 1847.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-02 09:45:00 | 1821.10 | 2025-09-03 09:15:00 | 1847.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-09-08 13:00:00 | 1814.20 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1809.20 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-09-10 10:00:00 | 1810.30 | 2025-09-16 09:15:00 | 1832.90 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-10-03 09:15:00 | 1815.80 | 2025-10-06 10:15:00 | 1788.80 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-17 15:15:00 | 1792.00 | 2025-10-30 10:15:00 | 1784.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-10-20 10:15:00 | 1785.80 | 2025-10-30 10:15:00 | 1784.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1789.80 | 2025-11-12 09:15:00 | 1792.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-11-17 09:45:00 | 1811.40 | 2025-11-17 12:15:00 | 1797.70 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 09:15:00 | 1686.60 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-11-28 13:15:00 | 1689.10 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-28 14:00:00 | 1689.70 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-01 09:45:00 | 1678.10 | 2025-12-02 12:15:00 | 1701.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-04 10:15:00 | 1735.10 | 2025-12-04 15:15:00 | 1689.20 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1658.50 | 2025-12-19 10:15:00 | 1575.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1658.00 | 2025-12-19 10:15:00 | 1575.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1657.70 | 2025-12-19 10:15:00 | 1574.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1658.50 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1658.00 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2025-12-10 14:00:00 | 1657.70 | 2025-12-19 14:15:00 | 1602.40 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2025-12-11 09:15:00 | 1653.30 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2025-12-12 10:15:00 | 1637.80 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.59% |
| SELL | retest2 | 2025-12-12 10:45:00 | 1637.00 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.55% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1638.40 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-12-15 09:45:00 | 1637.00 | 2025-12-22 12:15:00 | 1611.70 | STOP_HIT | 1.00 | 1.55% |
| BUY | retest2 | 2025-12-26 12:00:00 | 1620.00 | 2025-12-29 11:15:00 | 1611.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-31 09:45:00 | 1600.00 | 2025-12-31 13:15:00 | 1620.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1593.80 | 2026-01-12 09:15:00 | 1514.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:45:00 | 1593.80 | 2026-01-13 09:15:00 | 1526.80 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1593.90 | 2026-03-23 09:15:00 | 1540.04 | PARTIAL | 0.50 | 3.38% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1593.90 | 2026-03-24 11:15:00 | 1560.70 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2026-03-17 11:45:00 | 1621.10 | 2026-03-25 09:15:00 | 1620.40 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2026-04-09 09:15:00 | 1479.00 | 2026-04-15 10:15:00 | 1482.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1510.00 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2026-04-22 10:30:00 | 1509.30 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-04-22 12:15:00 | 1507.00 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-04-22 13:00:00 | 1507.30 | 2026-04-23 09:15:00 | 1493.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1483.00 | 2026-05-06 09:15:00 | 1408.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1487.20 | 2026-05-06 09:15:00 | 1412.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1486.90 | 2026-05-06 09:15:00 | 1412.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1483.60 | 2026-05-06 09:15:00 | 1409.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-28 15:15:00 | 1474.00 | 2026-05-06 09:15:00 | 1400.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-29 13:45:00 | 1472.40 | 2026-05-06 09:15:00 | 1398.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 11:15:00 | 1483.00 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.80% |
| SELL | retest2 | 2026-04-27 13:00:00 | 1487.20 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 3.07% |
| SELL | retest2 | 2026-04-27 13:45:00 | 1486.90 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2026-04-28 09:15:00 | 1483.60 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-04-28 15:15:00 | 1474.00 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2026-04-29 13:45:00 | 1472.40 | 2026-05-07 09:15:00 | 1441.50 | STOP_HIT | 0.50 | 2.10% |
