# Whirlpool of India Ltd. (WHIRLPOOL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 954.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 215 |
| ALERT1 | 132 |
| ALERT2 | 131 |
| ALERT2_SKIP | 88 |
| ALERT3 | 304 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 110 |
| PARTIAL | 18 |
| TARGET_HIT | 10 |
| STOP_HIT | 103 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 59 / 72
- **Target hits / Stop hits / Partials:** 10 / 103 / 18
- **Avg / median % per leg:** 1.01% / -0.28%
- **Sum % (uncompounded):** 131.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 13 | 25.5% | 7 | 44 | 0 | 0.23% | 11.6% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.22% | -4.4% |
| BUY @ 3rd Alert (retest2) | 49 | 13 | 26.5% | 7 | 42 | 0 | 0.33% | 16.1% |
| SELL (all) | 80 | 46 | 57.5% | 3 | 59 | 18 | 1.50% | 120.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 1 | 1 | 1 | 4.55% | 13.7% |
| SELL @ 3rd Alert (retest2) | 77 | 44 | 57.1% | 2 | 58 | 17 | 1.38% | 106.4% |
| retest1 (combined) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.84% | 9.2% |
| retest2 (combined) | 126 | 57 | 45.2% | 9 | 100 | 17 | 0.97% | 122.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 1420.00 | 1429.09 | 1429.81 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 11:15:00 | 1440.00 | 1430.24 | 1429.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-07 10:15:00 | 1452.85 | 1437.75 | 1434.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-07 15:15:00 | 1440.00 | 1443.78 | 1439.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 15:15:00 | 1440.00 | 1443.78 | 1439.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 15:15:00 | 1440.00 | 1443.78 | 1439.34 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-06-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 14:15:00 | 1432.70 | 1438.61 | 1438.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 12:15:00 | 1427.55 | 1433.52 | 1435.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 1432.45 | 1427.24 | 1431.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 1432.45 | 1427.24 | 1431.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 1432.45 | 1427.24 | 1431.50 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 14:15:00 | 1447.95 | 1434.10 | 1433.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 09:15:00 | 1472.80 | 1442.69 | 1437.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 1466.45 | 1467.49 | 1460.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 10:15:00 | 1458.00 | 1465.59 | 1460.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 1458.00 | 1465.59 | 1460.32 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 1445.95 | 1461.62 | 1463.59 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 1474.35 | 1461.58 | 1460.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 09:15:00 | 1478.45 | 1468.37 | 1464.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 1460.25 | 1469.70 | 1466.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 12:15:00 | 1460.25 | 1469.70 | 1466.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 1460.25 | 1469.70 | 1466.12 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 1446.35 | 1462.61 | 1463.66 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 1460.00 | 1457.05 | 1456.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 1460.70 | 1458.25 | 1457.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 13:15:00 | 1464.70 | 1465.58 | 1462.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 1461.25 | 1464.71 | 1461.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 1461.25 | 1464.71 | 1461.97 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 10:15:00 | 1462.90 | 1474.90 | 1475.70 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 14:15:00 | 1482.65 | 1475.55 | 1475.49 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 11:15:00 | 1473.05 | 1475.53 | 1475.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 15:15:00 | 1470.10 | 1473.47 | 1474.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 14:15:00 | 1453.45 | 1453.24 | 1459.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1453.00 | 1453.47 | 1458.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1453.00 | 1453.47 | 1458.81 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 1463.10 | 1461.14 | 1461.12 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 1453.70 | 1459.65 | 1460.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-13 10:15:00 | 1444.35 | 1451.75 | 1455.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 1444.00 | 1442.96 | 1448.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 10:15:00 | 1444.05 | 1443.17 | 1448.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 1444.05 | 1443.17 | 1448.20 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 10:15:00 | 1448.80 | 1438.45 | 1437.31 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 10:15:00 | 1431.10 | 1441.60 | 1442.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 10:15:00 | 1427.00 | 1437.07 | 1439.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 11:15:00 | 1427.00 | 1426.23 | 1431.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-28 09:15:00 | 1402.85 | 1414.02 | 1420.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 1402.85 | 1414.02 | 1420.17 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1448.30 | 1424.74 | 1421.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 10:15:00 | 1459.00 | 1431.59 | 1425.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 1449.20 | 1455.86 | 1448.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1449.20 | 1455.86 | 1448.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1449.20 | 1455.86 | 1448.67 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 1434.20 | 1445.78 | 1447.32 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 1455.00 | 1447.83 | 1446.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 1468.45 | 1451.95 | 1448.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-07 09:15:00 | 1448.45 | 1454.48 | 1451.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 1448.45 | 1454.48 | 1451.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 1448.45 | 1454.48 | 1451.06 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 12:15:00 | 1630.40 | 1635.04 | 1635.49 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 1639.40 | 1636.19 | 1635.89 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 1625.00 | 1633.95 | 1634.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-01 12:15:00 | 1614.50 | 1627.68 | 1631.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 14:15:00 | 1628.50 | 1627.01 | 1630.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 14:15:00 | 1628.50 | 1627.01 | 1630.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 1628.50 | 1627.01 | 1630.60 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 13:15:00 | 1650.45 | 1635.18 | 1633.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 1661.35 | 1644.76 | 1638.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 1656.95 | 1658.47 | 1650.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 09:15:00 | 1656.95 | 1658.47 | 1650.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1656.95 | 1658.47 | 1650.54 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 1670.00 | 1681.80 | 1681.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 1645.90 | 1674.62 | 1678.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 11:15:00 | 1653.35 | 1645.71 | 1657.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 11:15:00 | 1653.35 | 1645.71 | 1657.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 11:15:00 | 1653.35 | 1645.71 | 1657.56 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 10:15:00 | 1653.15 | 1642.06 | 1641.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-21 11:15:00 | 1656.25 | 1644.90 | 1642.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 09:15:00 | 1657.45 | 1659.45 | 1654.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 1657.45 | 1659.45 | 1654.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1657.45 | 1659.45 | 1654.95 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-09-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 12:15:00 | 1655.45 | 1658.90 | 1659.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 13:15:00 | 1648.25 | 1656.77 | 1658.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 15:15:00 | 1648.00 | 1647.60 | 1653.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-29 09:15:00 | 1660.00 | 1650.08 | 1654.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 1660.00 | 1650.08 | 1654.05 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 1654.90 | 1650.46 | 1650.10 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 13:15:00 | 1643.20 | 1649.38 | 1649.78 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 1653.00 | 1650.10 | 1650.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-05 15:15:00 | 1654.00 | 1650.88 | 1650.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-06 10:15:00 | 1649.35 | 1650.91 | 1650.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 10:15:00 | 1649.35 | 1650.91 | 1650.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 10:15:00 | 1649.35 | 1650.91 | 1650.54 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 13:15:00 | 1648.85 | 1650.16 | 1650.26 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 14:15:00 | 1652.10 | 1650.55 | 1650.43 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 09:15:00 | 1644.00 | 1649.31 | 1649.89 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 1655.00 | 1649.13 | 1648.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 1665.75 | 1655.01 | 1651.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 12:15:00 | 1663.45 | 1669.82 | 1664.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 12:15:00 | 1663.45 | 1669.82 | 1664.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 1663.45 | 1669.82 | 1664.15 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 13:15:00 | 1683.40 | 1692.24 | 1692.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 14:15:00 | 1679.85 | 1689.76 | 1691.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 13:15:00 | 1671.95 | 1671.53 | 1679.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 1636.35 | 1664.49 | 1675.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1636.35 | 1664.49 | 1675.93 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 15:15:00 | 1651.45 | 1641.24 | 1640.24 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 14:15:00 | 1633.55 | 1639.61 | 1640.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-30 15:15:00 | 1628.00 | 1637.28 | 1639.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 09:15:00 | 1640.80 | 1637.99 | 1639.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 09:15:00 | 1640.80 | 1637.99 | 1639.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 1640.80 | 1637.99 | 1639.39 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 11:15:00 | 1650.85 | 1641.79 | 1640.95 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 15:15:00 | 1625.15 | 1641.38 | 1641.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 09:15:00 | 1597.85 | 1619.51 | 1628.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 14:15:00 | 1607.10 | 1606.92 | 1617.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 12:15:00 | 1599.70 | 1599.19 | 1609.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 1599.70 | 1599.19 | 1609.46 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 09:15:00 | 1583.05 | 1576.54 | 1576.18 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 1568.55 | 1575.73 | 1576.05 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-11-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 13:15:00 | 1580.00 | 1576.58 | 1576.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 14:15:00 | 1590.00 | 1579.26 | 1577.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 15:15:00 | 1578.00 | 1579.01 | 1577.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 1583.80 | 1579.97 | 1578.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1583.80 | 1579.97 | 1578.24 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-13 15:15:00 | 1552.75 | 1577.32 | 1579.36 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 12:15:00 | 1582.55 | 1580.79 | 1580.67 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-11-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 13:15:00 | 1576.10 | 1579.85 | 1580.26 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 14:15:00 | 1600.35 | 1583.95 | 1582.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 1603.25 | 1590.41 | 1585.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 10:15:00 | 1603.15 | 1604.75 | 1597.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-17 11:15:00 | 1598.55 | 1603.51 | 1597.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 11:15:00 | 1598.55 | 1603.51 | 1597.51 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 09:15:00 | 1586.05 | 1594.31 | 1594.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 12:15:00 | 1576.80 | 1588.18 | 1591.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 14:15:00 | 1586.00 | 1585.92 | 1589.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 14:15:00 | 1586.00 | 1585.92 | 1589.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 1586.00 | 1585.92 | 1589.99 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 15:15:00 | 1580.00 | 1557.13 | 1554.95 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-01 09:15:00 | 1510.00 | 1547.70 | 1550.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-01 10:15:00 | 1497.05 | 1537.57 | 1545.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 14:15:00 | 1345.70 | 1342.48 | 1387.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 09:15:00 | 1325.00 | 1314.40 | 1325.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 1325.00 | 1314.40 | 1325.91 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 09:15:00 | 1348.10 | 1331.40 | 1329.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 10:15:00 | 1360.00 | 1342.94 | 1338.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 12:15:00 | 1342.55 | 1343.99 | 1339.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 13:15:00 | 1338.10 | 1342.81 | 1339.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 13:15:00 | 1338.10 | 1342.81 | 1339.72 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-21 12:15:00 | 1350.00 | 1355.11 | 1355.77 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 1364.60 | 1357.43 | 1356.59 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 11:15:00 | 1351.05 | 1358.66 | 1359.60 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-12-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 14:15:00 | 1361.60 | 1358.72 | 1358.62 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-29 09:15:00 | 1354.00 | 1358.00 | 1358.32 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2023-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 11:15:00 | 1361.80 | 1358.92 | 1358.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 14:15:00 | 1365.00 | 1360.64 | 1359.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 10:15:00 | 1361.65 | 1362.45 | 1360.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 11:15:00 | 1367.30 | 1363.42 | 1361.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 1367.30 | 1363.42 | 1361.42 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 10:15:00 | 1355.65 | 1361.16 | 1361.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 11:15:00 | 1352.00 | 1359.33 | 1360.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 09:15:00 | 1359.55 | 1357.37 | 1358.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 09:15:00 | 1359.55 | 1357.37 | 1358.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 1359.55 | 1357.37 | 1358.76 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 12:15:00 | 1362.85 | 1359.59 | 1359.52 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-01-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-04 10:15:00 | 1353.70 | 1359.87 | 1359.94 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 1366.10 | 1360.54 | 1359.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 09:15:00 | 1369.80 | 1363.41 | 1361.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 14:15:00 | 1362.15 | 1365.29 | 1363.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-08 14:15:00 | 1362.15 | 1365.29 | 1363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 1362.15 | 1365.29 | 1363.49 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-10 10:15:00 | 1356.85 | 1363.19 | 1363.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 13:15:00 | 1355.15 | 1359.43 | 1361.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 14:15:00 | 1355.75 | 1354.77 | 1357.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 14:15:00 | 1355.75 | 1354.77 | 1357.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 1355.75 | 1354.77 | 1357.46 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 15:15:00 | 1359.40 | 1357.14 | 1356.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-16 10:15:00 | 1364.70 | 1359.61 | 1358.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 11:15:00 | 1358.65 | 1359.42 | 1358.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 11:15:00 | 1358.65 | 1359.42 | 1358.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 1358.65 | 1359.42 | 1358.13 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-19 12:15:00 | 1364.50 | 1371.80 | 1372.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-19 13:15:00 | 1362.40 | 1369.92 | 1371.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 1330.00 | 1326.79 | 1339.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 14:15:00 | 1338.20 | 1330.49 | 1337.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1338.20 | 1330.49 | 1337.41 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 11:15:00 | 1340.40 | 1326.23 | 1325.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 12:15:00 | 1345.05 | 1329.99 | 1327.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 1338.90 | 1339.36 | 1334.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 09:15:00 | 1357.85 | 1347.28 | 1342.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 1357.85 | 1347.28 | 1342.57 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 1353.30 | 1361.68 | 1362.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 1346.55 | 1356.70 | 1359.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 1346.10 | 1343.47 | 1348.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 1346.10 | 1343.47 | 1348.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 1346.10 | 1343.47 | 1348.75 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 11:15:00 | 1265.00 | 1260.54 | 1260.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 13:15:00 | 1282.05 | 1265.69 | 1262.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 10:15:00 | 1263.65 | 1270.76 | 1266.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 10:15:00 | 1263.65 | 1270.76 | 1266.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 1263.65 | 1270.76 | 1266.85 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 15:15:00 | 1260.15 | 1264.68 | 1265.16 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 09:15:00 | 1277.15 | 1267.17 | 1266.25 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 14:15:00 | 1238.90 | 1261.75 | 1264.31 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 1260.10 | 1249.40 | 1249.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 12:15:00 | 1261.95 | 1251.91 | 1250.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 09:15:00 | 1246.65 | 1254.22 | 1252.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 09:15:00 | 1246.65 | 1254.22 | 1252.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 1246.65 | 1254.22 | 1252.40 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 13:15:00 | 1251.25 | 1253.55 | 1253.56 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 14:15:00 | 1256.70 | 1254.18 | 1253.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-12 15:15:00 | 1257.30 | 1254.80 | 1254.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1250.25 | 1253.89 | 1253.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 1250.25 | 1253.89 | 1253.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1250.25 | 1253.89 | 1253.81 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 10:15:00 | 1245.80 | 1253.42 | 1254.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 13:15:00 | 1236.90 | 1246.88 | 1250.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 1255.15 | 1248.54 | 1251.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 14:15:00 | 1255.15 | 1248.54 | 1251.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 1255.15 | 1248.54 | 1251.24 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 1260.50 | 1247.06 | 1246.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 13:15:00 | 1263.20 | 1252.36 | 1249.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 1234.80 | 1260.47 | 1258.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 1234.80 | 1260.47 | 1258.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1234.80 | 1260.47 | 1258.01 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-20 11:15:00 | 1251.75 | 1256.40 | 1256.45 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 14:15:00 | 1262.45 | 1257.23 | 1256.75 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 10:15:00 | 1253.25 | 1256.08 | 1256.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 09:15:00 | 1243.90 | 1252.48 | 1254.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-27 13:15:00 | 1225.25 | 1222.71 | 1229.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 15:15:00 | 1236.40 | 1225.64 | 1229.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 15:15:00 | 1236.40 | 1225.64 | 1229.43 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 1238.10 | 1228.90 | 1228.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 1240.70 | 1231.26 | 1229.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 1401.55 | 1404.55 | 1384.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 15:15:00 | 1405.20 | 1404.29 | 1388.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 1405.20 | 1404.29 | 1388.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1426.55 | 1419.79 | 1406.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-12 15:15:00 | 1426.10 | 1424.45 | 1414.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 10:30:00 | 1425.00 | 1421.01 | 1415.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-15 13:00:00 | 1422.15 | 1421.40 | 1416.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 1418.35 | 1421.37 | 1417.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 1418.35 | 1421.37 | 1417.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 1411.00 | 1419.29 | 1416.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 1425.00 | 1419.29 | 1416.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1428.45 | 1421.12 | 1417.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 15:00:00 | 1440.00 | 1427.21 | 1423.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-23 09:15:00 | 1569.21 | 1515.36 | 1480.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 15:15:00 | 1517.00 | 1522.51 | 1522.93 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 1530.85 | 1524.18 | 1523.65 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 10:15:00 | 1517.10 | 1522.76 | 1523.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 1511.90 | 1520.59 | 1522.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 13:15:00 | 1494.40 | 1494.27 | 1504.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-29 13:45:00 | 1493.50 | 1494.27 | 1504.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 14:15:00 | 1501.45 | 1486.77 | 1493.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 15:00:00 | 1501.45 | 1486.77 | 1493.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 15:15:00 | 1504.80 | 1490.37 | 1494.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 10:00:00 | 1489.50 | 1490.20 | 1493.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 1486.00 | 1489.84 | 1492.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 09:30:00 | 1488.75 | 1485.97 | 1489.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 1415.02 | 1428.94 | 1446.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 1411.70 | 1428.94 | 1446.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 11:15:00 | 1414.31 | 1428.94 | 1446.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-07 13:15:00 | 1438.45 | 1429.41 | 1443.42 | SL hit (close>ema200) qty=0.50 sl=1429.41 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 1439.00 | 1429.60 | 1429.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 1457.95 | 1442.75 | 1436.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1567.50 | 1571.70 | 1548.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1567.50 | 1571.70 | 1548.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1540.90 | 1563.03 | 1548.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1548.00 | 1563.03 | 1548.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1540.50 | 1558.52 | 1547.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 1540.00 | 1558.52 | 1547.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 1521.25 | 1540.19 | 1541.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 10:15:00 | 1504.25 | 1526.46 | 1533.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 14:15:00 | 1528.55 | 1519.98 | 1527.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 1528.55 | 1519.98 | 1527.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1528.55 | 1519.98 | 1527.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1528.55 | 1519.98 | 1527.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1526.00 | 1521.18 | 1527.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 1544.40 | 1521.18 | 1527.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1528.20 | 1522.59 | 1527.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 1524.30 | 1523.69 | 1526.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 13:15:00 | 1524.05 | 1524.45 | 1526.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 1525.05 | 1525.43 | 1526.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 1538.90 | 1528.12 | 1527.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 1538.90 | 1528.12 | 1527.80 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1491.35 | 1524.57 | 1528.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 1472.75 | 1514.20 | 1523.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 12:15:00 | 1480.00 | 1477.22 | 1493.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 12:45:00 | 1474.90 | 1477.22 | 1493.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1524.75 | 1487.08 | 1494.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 1524.75 | 1487.08 | 1494.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 1495.00 | 1488.66 | 1494.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 1519.90 | 1488.66 | 1494.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1501.55 | 1491.24 | 1495.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:15:00 | 1492.90 | 1491.24 | 1495.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:00:00 | 1494.70 | 1494.22 | 1496.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 14:15:00 | 1514.45 | 1500.43 | 1498.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 14:15:00 | 1514.45 | 1500.43 | 1498.73 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1442.10 | 1493.03 | 1496.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1425.70 | 1479.57 | 1489.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1475.15 | 1474.27 | 1485.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 14:00:00 | 1475.15 | 1474.27 | 1485.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1489.30 | 1475.77 | 1483.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 1485.05 | 1475.77 | 1483.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 1510.00 | 1482.62 | 1486.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:00:00 | 1510.00 | 1482.62 | 1486.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1542.65 | 1494.62 | 1491.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1548.60 | 1530.11 | 1513.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 14:15:00 | 1799.10 | 1799.12 | 1764.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:30:00 | 1800.05 | 1799.12 | 1764.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1760.80 | 1784.43 | 1775.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:30:00 | 1794.60 | 1784.38 | 1777.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 12:15:00 | 1833.00 | 1838.06 | 1838.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 12:15:00 | 1833.00 | 1838.06 | 1838.60 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 15:15:00 | 1844.00 | 1838.79 | 1838.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 2011.75 | 1873.39 | 1854.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 12:15:00 | 1981.15 | 1986.56 | 1962.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 13:00:00 | 1981.15 | 1986.56 | 1962.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 1958.40 | 1980.93 | 1962.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 1958.40 | 1980.93 | 1962.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 1959.50 | 1976.64 | 1962.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:45:00 | 1958.80 | 1976.64 | 1962.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 1957.00 | 1972.71 | 1961.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 09:15:00 | 1965.00 | 1972.71 | 1961.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 1934.85 | 1965.14 | 1959.34 | SL hit (close<static) qty=1.00 sl=1947.95 alert=retest2 |

### Cycle 89 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 1945.00 | 1954.76 | 1955.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 15:15:00 | 1932.05 | 1946.07 | 1950.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 13:15:00 | 1942.25 | 1939.52 | 1945.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 1942.25 | 1939.52 | 1945.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 1942.25 | 1939.52 | 1945.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 1939.50 | 1939.52 | 1945.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1934.60 | 1938.54 | 1944.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:15:00 | 1950.00 | 1938.54 | 1944.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1950.00 | 1940.83 | 1944.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1964.00 | 1940.83 | 1944.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1970.95 | 1946.86 | 1947.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 1967.25 | 1946.86 | 1947.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 10:15:00 | 1962.70 | 1950.02 | 1948.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 13:15:00 | 2001.95 | 1962.26 | 1954.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-05 09:15:00 | 1975.00 | 1976.45 | 1963.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-05 09:15:00 | 1975.00 | 1976.45 | 1963.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1975.00 | 1976.45 | 1963.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 1968.00 | 1976.45 | 1963.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1992.70 | 1998.76 | 1982.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:30:00 | 1988.00 | 1998.76 | 1982.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1987.65 | 1996.54 | 1983.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1987.65 | 1996.54 | 1983.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 1990.00 | 1995.23 | 1983.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:30:00 | 1981.20 | 1995.23 | 1983.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1978.90 | 1994.52 | 1988.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:00:00 | 1978.90 | 1994.52 | 1988.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1982.45 | 1992.11 | 1987.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 12:15:00 | 2006.90 | 1990.85 | 1987.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 15:15:00 | 2000.00 | 2012.16 | 2012.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 15:15:00 | 2000.00 | 2012.16 | 2012.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 1989.00 | 2007.53 | 2010.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 10:15:00 | 2003.00 | 1990.50 | 1998.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 10:15:00 | 2003.00 | 1990.50 | 1998.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 10:15:00 | 2003.00 | 1990.50 | 1998.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 10:30:00 | 2011.00 | 1990.50 | 1998.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 11:15:00 | 1999.00 | 1992.20 | 1998.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 11:45:00 | 2010.00 | 1992.20 | 1998.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 2019.75 | 2000.76 | 2000.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 2019.75 | 2000.76 | 2000.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 2022.00 | 2005.01 | 2002.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 2035.00 | 2012.96 | 2006.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 2026.60 | 2039.41 | 2027.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 2026.60 | 2039.41 | 2027.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 2026.60 | 2039.41 | 2027.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 2026.60 | 2039.41 | 2027.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 2029.65 | 2037.46 | 2027.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:15:00 | 2043.70 | 2037.46 | 2027.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:30:00 | 2039.05 | 2037.66 | 2030.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 2015.00 | 2026.21 | 2026.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 2015.00 | 2026.21 | 2026.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 2005.45 | 2020.26 | 2023.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 11:15:00 | 2015.00 | 2009.81 | 2016.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 11:15:00 | 2015.00 | 2009.81 | 2016.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 2015.00 | 2009.81 | 2016.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 2015.00 | 2009.81 | 2016.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 2025.35 | 2012.91 | 2017.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 2025.35 | 2012.91 | 2017.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 2038.00 | 2017.93 | 2019.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 2036.40 | 2017.93 | 2019.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 2044.40 | 2023.23 | 2021.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 2053.10 | 2033.12 | 2026.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 15:15:00 | 2039.60 | 2040.43 | 2032.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 15:15:00 | 2039.60 | 2040.43 | 2032.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 2039.60 | 2040.43 | 2032.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 2009.75 | 2040.43 | 2032.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1995.00 | 2031.34 | 2029.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 1995.75 | 2031.34 | 2029.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 10:15:00 | 1990.85 | 2023.24 | 2025.89 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 11:15:00 | 2052.00 | 2025.33 | 2022.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 2060.00 | 2042.88 | 2034.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 2121.70 | 2133.45 | 2110.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 2121.70 | 2133.45 | 2110.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 2121.70 | 2133.45 | 2110.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 2121.70 | 2133.45 | 2110.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 2117.70 | 2131.87 | 2123.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:00:00 | 2117.70 | 2131.87 | 2123.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 2102.10 | 2125.91 | 2121.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 2106.70 | 2125.91 | 2121.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 2122.50 | 2125.23 | 2121.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 13:15:00 | 2131.65 | 2126.18 | 2122.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:30:00 | 2142.55 | 2127.21 | 2123.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:15:00 | 2148.00 | 2127.21 | 2123.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 11:45:00 | 2130.60 | 2127.54 | 2125.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 2120.10 | 2126.05 | 2124.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:45:00 | 2120.00 | 2126.05 | 2124.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 2126.35 | 2126.11 | 2124.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:30:00 | 2120.05 | 2126.11 | 2124.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 2130.50 | 2126.99 | 2125.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 14:30:00 | 2132.50 | 2126.99 | 2125.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 2070.00 | 2116.39 | 2120.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 2070.00 | 2116.39 | 2120.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 2027.30 | 2098.57 | 2112.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2071.60 | 2054.74 | 2079.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 2071.60 | 2054.74 | 2079.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 2075.00 | 2058.79 | 2078.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:00:00 | 2075.00 | 2058.79 | 2078.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 11:15:00 | 2073.15 | 2061.66 | 2078.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 11:30:00 | 2073.80 | 2061.66 | 2078.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 2053.45 | 2045.62 | 2059.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:45:00 | 2054.45 | 2045.62 | 2059.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 2072.00 | 2050.90 | 2060.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 2072.00 | 2050.90 | 2060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 2068.20 | 2054.36 | 2060.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:15:00 | 2059.55 | 2054.36 | 2060.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 2059.55 | 2055.40 | 2060.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 2082.00 | 2060.72 | 2062.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 2076.05 | 2063.78 | 2063.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:30:00 | 2086.00 | 2063.78 | 2063.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 2082.25 | 2067.48 | 2065.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 14:15:00 | 2098.70 | 2074.77 | 2069.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 2090.40 | 2091.29 | 2082.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:30:00 | 2093.15 | 2091.29 | 2082.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 2081.40 | 2089.32 | 2082.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 2095.05 | 2089.32 | 2082.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 2080.70 | 2087.59 | 2081.94 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 2066.70 | 2079.20 | 2080.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 2054.50 | 2074.26 | 2078.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 12:15:00 | 2074.60 | 2056.81 | 2065.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 12:15:00 | 2074.60 | 2056.81 | 2065.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 2074.60 | 2056.81 | 2065.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:45:00 | 2076.30 | 2056.81 | 2065.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 13:15:00 | 2075.70 | 2060.59 | 2066.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 13:45:00 | 2075.05 | 2060.59 | 2066.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2066.80 | 2065.91 | 2067.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 2075.00 | 2065.91 | 2067.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2075.05 | 2067.74 | 2068.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 2075.05 | 2067.74 | 2068.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 2121.00 | 2078.39 | 2073.30 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 14:15:00 | 2063.70 | 2084.30 | 2085.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 09:15:00 | 2052.75 | 2074.25 | 2080.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 11:15:00 | 2082.20 | 2072.92 | 2078.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 11:15:00 | 2082.20 | 2072.92 | 2078.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 11:15:00 | 2082.20 | 2072.92 | 2078.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:00:00 | 2082.20 | 2072.92 | 2078.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 2082.00 | 2074.74 | 2078.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-21 13:15:00 | 2072.30 | 2074.74 | 2078.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 09:30:00 | 2067.80 | 2071.13 | 2075.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 2071.65 | 2070.26 | 2074.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 2047.95 | 2037.56 | 2037.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 2047.95 | 2037.56 | 2037.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 14:15:00 | 2080.00 | 2048.48 | 2042.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 15:15:00 | 2172.00 | 2172.54 | 2141.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 09:15:00 | 2163.00 | 2172.54 | 2141.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 2155.05 | 2169.05 | 2142.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:45:00 | 2152.70 | 2169.05 | 2142.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 2203.60 | 2218.40 | 2197.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:45:00 | 2201.40 | 2218.40 | 2197.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2199.15 | 2212.89 | 2198.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 11:30:00 | 2200.65 | 2207.81 | 2198.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 2222.80 | 2204.98 | 2198.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:15:00 | 2208.55 | 2210.38 | 2205.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 2178.15 | 2211.63 | 2212.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 2178.15 | 2211.63 | 2212.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 12:15:00 | 2155.60 | 2190.56 | 2201.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 15:15:00 | 2123.00 | 2120.59 | 2140.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-12 09:15:00 | 2107.35 | 2120.59 | 2140.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 2105.95 | 2117.66 | 2137.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 2090.05 | 2103.52 | 2114.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:15:00 | 1985.55 | 2056.31 | 2083.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 2070.85 | 2043.22 | 2060.52 | SL hit (close>ema200) qty=0.50 sl=2043.22 alert=retest2 |

### Cycle 104 — BUY (started 2024-09-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 15:15:00 | 2069.00 | 2049.73 | 2047.43 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 09:15:00 | 2045.60 | 2048.28 | 2048.55 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 2065.95 | 2051.81 | 2050.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 12:15:00 | 2071.40 | 2056.74 | 2052.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 2039.95 | 2053.38 | 2051.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 13:15:00 | 2039.95 | 2053.38 | 2051.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 2039.95 | 2053.38 | 2051.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 2039.95 | 2053.38 | 2051.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 2070.00 | 2056.71 | 2053.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 2041.50 | 2056.71 | 2053.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 2063.00 | 2058.33 | 2054.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 2063.00 | 2058.33 | 2054.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 2065.35 | 2059.74 | 2055.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 2119.90 | 2072.03 | 2062.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-30 11:15:00 | 2331.89 | 2228.08 | 2178.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 2349.25 | 2367.25 | 2369.34 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 14:15:00 | 2431.60 | 2380.44 | 2374.59 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 15:15:00 | 2352.80 | 2375.09 | 2376.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 2309.80 | 2362.03 | 2370.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 2360.25 | 2349.70 | 2358.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 2360.25 | 2349.70 | 2358.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2360.25 | 2349.70 | 2358.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 2347.55 | 2349.70 | 2358.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 2364.65 | 2352.69 | 2358.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 2364.65 | 2352.69 | 2358.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 2350.00 | 2352.15 | 2358.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:45:00 | 2374.05 | 2352.15 | 2358.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2350.00 | 2351.78 | 2356.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 2352.40 | 2351.78 | 2356.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 2377.95 | 2353.77 | 2356.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:30:00 | 2381.80 | 2353.77 | 2356.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 10:15:00 | 2395.55 | 2362.13 | 2359.91 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 2349.00 | 2360.53 | 2360.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 2312.50 | 2350.92 | 2356.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 18:15:00 | 2045.00 | 2020.54 | 2051.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 18:15:00 | 2045.00 | 2020.54 | 2051.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 2045.00 | 2020.54 | 2051.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:45:00 | 2045.00 | 2020.54 | 2051.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 2035.65 | 2007.74 | 2025.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 2035.65 | 2007.74 | 2025.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 2047.50 | 2015.69 | 2027.82 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 2049.65 | 2033.96 | 2033.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 10:15:00 | 2058.40 | 2038.85 | 2035.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 13:15:00 | 2042.30 | 2043.01 | 2038.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 2042.30 | 2043.01 | 2038.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 2042.30 | 2043.01 | 2038.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:30:00 | 2038.85 | 2043.01 | 2038.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2038.30 | 2045.76 | 2041.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2038.30 | 2045.76 | 2041.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2046.90 | 2045.99 | 2041.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2035.35 | 2045.99 | 2041.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2061.60 | 2049.11 | 2043.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:30:00 | 2079.55 | 2057.77 | 2047.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 13:15:00 | 2039.00 | 2055.19 | 2054.21 | SL hit (close<static) qty=1.00 sl=2041.80 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 09:15:00 | 1973.25 | 2041.37 | 2048.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 1947.15 | 2010.78 | 2032.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1835.25 | 1790.36 | 1839.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1835.25 | 1790.36 | 1839.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1849.85 | 1802.26 | 1840.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:00:00 | 1849.85 | 1802.26 | 1840.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1848.15 | 1811.44 | 1841.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 12:45:00 | 1837.30 | 1817.99 | 1841.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 1834.15 | 1815.63 | 1838.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 11:30:00 | 1822.00 | 1819.26 | 1822.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:30:00 | 1834.00 | 1823.01 | 1823.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 14:15:00 | 1835.00 | 1825.41 | 1824.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 14:15:00 | 1835.00 | 1825.41 | 1824.87 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 09:15:00 | 1784.00 | 1818.98 | 1822.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 13:15:00 | 1770.45 | 1795.36 | 1808.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 10:15:00 | 1767.65 | 1766.22 | 1788.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 11:00:00 | 1767.65 | 1766.22 | 1788.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1849.85 | 1782.41 | 1785.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 10:00:00 | 1849.85 | 1782.41 | 1785.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1800.45 | 1786.02 | 1787.26 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 1801.95 | 1789.20 | 1788.60 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 09:15:00 | 1779.05 | 1787.16 | 1788.19 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 11:15:00 | 1797.65 | 1789.73 | 1789.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 12:15:00 | 1799.90 | 1791.76 | 1790.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 1790.45 | 1798.63 | 1794.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 09:15:00 | 1790.45 | 1798.63 | 1794.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1790.45 | 1798.63 | 1794.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 1790.45 | 1798.63 | 1794.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1793.45 | 1797.60 | 1794.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:15:00 | 1799.85 | 1796.51 | 1794.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 1811.25 | 1797.04 | 1795.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 14:15:00 | 1884.90 | 1887.82 | 1888.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 1884.90 | 1887.82 | 1888.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 15:15:00 | 1870.00 | 1884.25 | 1886.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1898.80 | 1887.16 | 1887.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 1898.80 | 1887.16 | 1887.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 1898.80 | 1887.16 | 1887.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:30:00 | 1914.60 | 1887.16 | 1887.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 10:15:00 | 1905.55 | 1890.84 | 1889.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 11:15:00 | 1916.40 | 1895.95 | 1891.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 1899.00 | 1910.48 | 1901.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 1899.00 | 1910.48 | 1901.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1899.00 | 1910.48 | 1901.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:00:00 | 1899.00 | 1910.48 | 1901.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 1894.45 | 1907.28 | 1901.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:00:00 | 1894.45 | 1907.28 | 1901.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 1902.65 | 1906.35 | 1901.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 12:30:00 | 1912.75 | 1907.84 | 1902.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 1911.00 | 1922.89 | 1919.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 15:15:00 | 1904.80 | 1916.72 | 1917.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 15:15:00 | 1904.80 | 1916.72 | 1917.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 1897.65 | 1909.01 | 1913.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1912.55 | 1909.72 | 1913.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 1912.55 | 1909.72 | 1913.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1912.55 | 1909.72 | 1913.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:45:00 | 1910.90 | 1909.72 | 1913.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1914.65 | 1910.71 | 1913.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 1913.45 | 1910.71 | 1913.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 1909.65 | 1910.49 | 1913.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 14:45:00 | 1915.15 | 1910.49 | 1913.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 1909.10 | 1910.22 | 1912.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 1917.60 | 1910.22 | 1912.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1916.50 | 1911.47 | 1913.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 1916.75 | 1911.47 | 1913.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 1921.70 | 1913.52 | 1914.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 11:00:00 | 1921.70 | 1913.52 | 1914.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 1926.70 | 1916.15 | 1915.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 14:15:00 | 1946.90 | 1923.67 | 1918.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1926.90 | 1931.08 | 1924.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:00:00 | 1926.90 | 1931.08 | 1924.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 1946.80 | 1934.22 | 1926.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 13:45:00 | 1955.30 | 1939.75 | 1933.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 14:45:00 | 1961.15 | 1941.26 | 1934.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:15:00 | 1962.15 | 1941.26 | 1934.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1958.25 | 1948.28 | 1939.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 1956.75 | 1956.75 | 1946.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:45:00 | 1955.05 | 1956.75 | 1946.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1934.85 | 1952.37 | 1945.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 1934.85 | 1952.37 | 1945.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1934.85 | 1948.87 | 1944.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:15:00 | 1915.10 | 1948.87 | 1944.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1915.10 | 1942.11 | 1941.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 15:15:00 | 1915.10 | 1942.11 | 1941.51 | SL hit (close<static) qty=1.00 sl=1923.35 alert=retest2 |

### Cycle 123 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 1902.10 | 1934.11 | 1937.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 11:15:00 | 1875.00 | 1916.67 | 1928.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1887.55 | 1880.53 | 1903.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1887.55 | 1880.53 | 1903.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1887.55 | 1880.53 | 1903.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:45:00 | 1895.00 | 1880.53 | 1903.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1891.90 | 1880.25 | 1897.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:45:00 | 1897.95 | 1880.25 | 1897.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1898.00 | 1883.80 | 1897.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:45:00 | 1892.60 | 1883.80 | 1897.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 1889.10 | 1884.86 | 1896.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 15:15:00 | 1878.30 | 1884.86 | 1896.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1878.30 | 1883.55 | 1894.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1870.05 | 1883.55 | 1894.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:30:00 | 1874.75 | 1869.73 | 1879.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:15:00 | 1877.65 | 1882.34 | 1883.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 15:00:00 | 1871.90 | 1880.25 | 1882.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 1862.00 | 1876.60 | 1880.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1850.00 | 1876.60 | 1880.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:45:00 | 1860.95 | 1873.36 | 1878.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 12:00:00 | 1859.75 | 1869.39 | 1875.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 1838.20 | 1869.49 | 1875.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1808.30 | 1857.25 | 1869.16 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 1781.01 | 1843.41 | 1861.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 1783.77 | 1843.41 | 1861.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-30 14:15:00 | 1778.31 | 1843.41 | 1861.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 14:30:00 | 1790.25 | 1843.41 | 1861.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1795.50 | 1835.40 | 1856.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-31 14:15:00 | 1836.70 | 1824.16 | 1839.87 | SL hit (close>ema200) qty=0.50 sl=1824.16 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 14:15:00 | 1848.70 | 1836.97 | 1835.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 10:15:00 | 1860.90 | 1843.06 | 1838.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1851.65 | 1861.48 | 1851.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1851.65 | 1861.48 | 1851.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1851.65 | 1861.48 | 1851.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 1857.50 | 1861.48 | 1851.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1829.50 | 1855.08 | 1849.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1829.50 | 1855.08 | 1849.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1813.80 | 1846.83 | 1846.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1811.80 | 1846.83 | 1846.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 1809.15 | 1839.29 | 1843.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1799.00 | 1831.23 | 1839.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1729.25 | 1727.51 | 1757.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 1729.25 | 1727.51 | 1757.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 1720.00 | 1707.12 | 1719.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 1674.90 | 1707.12 | 1719.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 14:15:00 | 1591.15 | 1640.31 | 1676.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 1640.05 | 1627.76 | 1657.94 | SL hit (close>ema200) qty=0.50 sl=1627.76 alert=retest2 |

### Cycle 126 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 1695.80 | 1671.09 | 1669.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 1717.25 | 1691.57 | 1682.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 13:15:00 | 1688.10 | 1692.45 | 1684.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 13:15:00 | 1688.10 | 1692.45 | 1684.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 1688.10 | 1692.45 | 1684.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:00:00 | 1688.10 | 1692.45 | 1684.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 1670.00 | 1687.96 | 1682.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 1670.00 | 1687.96 | 1682.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 1671.90 | 1684.75 | 1681.92 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 1664.20 | 1679.54 | 1680.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 12:15:00 | 1656.50 | 1674.93 | 1678.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 14:15:00 | 1681.30 | 1675.46 | 1677.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 14:15:00 | 1681.30 | 1675.46 | 1677.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1681.30 | 1675.46 | 1677.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:45:00 | 1685.00 | 1675.46 | 1677.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 1685.00 | 1677.37 | 1678.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 1670.95 | 1677.37 | 1678.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:45:00 | 1666.70 | 1676.08 | 1677.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1587.40 | 1619.63 | 1641.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 1583.37 | 1619.63 | 1641.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-23 09:15:00 | 1600.85 | 1590.33 | 1612.14 | SL hit (close>ema200) qty=0.50 sl=1590.33 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1002.05 | 996.41 | 996.23 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 10:15:00 | 988.55 | 995.14 | 995.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 11:15:00 | 981.65 | 992.44 | 994.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 990.05 | 989.06 | 992.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 990.05 | 989.06 | 992.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 976.30 | 986.34 | 990.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:30:00 | 988.90 | 986.34 | 990.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 965.20 | 977.57 | 983.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 10:45:00 | 954.30 | 964.51 | 969.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 945.00 | 960.98 | 965.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 906.58 | 925.10 | 940.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 10:15:00 | 927.65 | 925.61 | 939.60 | SL hit (close>ema200) qty=0.50 sl=925.61 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 15:15:00 | 942.00 | 934.95 | 934.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 950.55 | 938.88 | 936.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 964.55 | 974.04 | 962.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 999.65 | 974.04 | 962.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 989.00 | 993.82 | 982.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 975.70 | 993.82 | 982.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 978.10 | 990.68 | 981.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 10:15:00 | 978.10 | 990.68 | 981.66 | SL hit (close<ema400) qty=1.00 sl=981.66 alert=retest1 |

### Cycle 131 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 962.65 | 977.54 | 977.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 961.95 | 974.43 | 976.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 940.00 | 937.20 | 947.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 940.00 | 937.20 | 947.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 945.20 | 939.15 | 946.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:45:00 | 940.10 | 939.15 | 946.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 945.95 | 940.51 | 946.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 939.80 | 944.21 | 946.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 14:00:00 | 939.85 | 938.81 | 942.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 934.00 | 937.85 | 941.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 968.00 | 943.75 | 943.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 968.00 | 943.75 | 943.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 973.85 | 955.63 | 950.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 12:15:00 | 982.00 | 982.20 | 974.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 12:30:00 | 980.00 | 982.20 | 974.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1009.55 | 988.26 | 979.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:15:00 | 1010.00 | 988.26 | 979.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 984.10 | 988.25 | 988.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 984.10 | 988.25 | 988.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 970.15 | 984.63 | 987.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 09:15:00 | 975.00 | 973.78 | 979.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 09:15:00 | 975.00 | 973.78 | 979.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 975.00 | 973.78 | 979.18 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 990.00 | 980.72 | 980.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 09:15:00 | 1010.30 | 986.64 | 983.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 986.65 | 993.89 | 989.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 986.65 | 993.89 | 989.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 986.65 | 993.89 | 989.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 986.65 | 993.89 | 989.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 993.00 | 993.71 | 989.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 998.30 | 993.71 | 989.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1008.80 | 996.73 | 991.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:00:00 | 1019.20 | 1002.09 | 994.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 986.50 | 1045.55 | 1049.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 986.50 | 1045.55 | 1049.85 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1060.05 | 1045.18 | 1043.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 13:15:00 | 1067.55 | 1052.05 | 1047.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 13:15:00 | 1060.00 | 1065.38 | 1058.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 13:15:00 | 1060.00 | 1065.38 | 1058.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 13:15:00 | 1060.00 | 1065.38 | 1058.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 14:00:00 | 1060.00 | 1065.38 | 1058.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 1072.75 | 1067.59 | 1060.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 1077.15 | 1067.59 | 1060.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 1094.00 | 1099.35 | 1099.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 11:15:00 | 1094.00 | 1099.35 | 1099.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-22 09:15:00 | 1080.20 | 1093.27 | 1096.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 10:15:00 | 1102.25 | 1095.07 | 1097.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 10:15:00 | 1102.25 | 1095.07 | 1097.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 1102.25 | 1095.07 | 1097.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 1102.25 | 1095.07 | 1097.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 1105.75 | 1097.20 | 1097.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:45:00 | 1105.00 | 1097.20 | 1097.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 1110.05 | 1099.77 | 1098.96 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 1088.00 | 1098.08 | 1098.96 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 09:15:00 | 1124.45 | 1098.72 | 1098.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 10:15:00 | 1138.90 | 1106.76 | 1101.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 1143.35 | 1146.68 | 1126.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-25 10:00:00 | 1143.35 | 1146.68 | 1126.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1140.40 | 1145.43 | 1128.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1140.40 | 1145.43 | 1128.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 1268.10 | 1265.39 | 1239.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 12:30:00 | 1240.00 | 1265.39 | 1239.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 1251.15 | 1262.10 | 1244.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 1255.30 | 1262.10 | 1244.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1244.20 | 1258.52 | 1244.74 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 09:15:00 | 1238.00 | 1239.43 | 1239.46 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 1255.10 | 1242.56 | 1240.88 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 1230.00 | 1238.52 | 1239.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 1201.50 | 1228.39 | 1234.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 09:15:00 | 1212.80 | 1209.91 | 1220.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 1212.80 | 1209.91 | 1220.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 1212.80 | 1209.91 | 1220.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:30:00 | 1215.00 | 1209.91 | 1220.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1238.40 | 1215.61 | 1221.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 1232.60 | 1215.61 | 1221.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1253.00 | 1223.09 | 1224.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 1253.00 | 1223.09 | 1224.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 14:15:00 | 1237.30 | 1227.87 | 1226.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 1249.00 | 1233.73 | 1229.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 1232.60 | 1240.01 | 1234.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 1232.60 | 1240.01 | 1234.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1232.60 | 1240.01 | 1234.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1232.60 | 1240.01 | 1234.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1223.00 | 1236.61 | 1233.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1209.60 | 1236.61 | 1233.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1225.00 | 1234.28 | 1233.05 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1221.20 | 1231.67 | 1231.97 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1279.60 | 1235.40 | 1232.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 10:15:00 | 1288.50 | 1269.77 | 1255.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1262.30 | 1269.14 | 1257.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1262.30 | 1269.14 | 1257.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1262.30 | 1269.14 | 1257.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1262.30 | 1269.14 | 1257.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1267.00 | 1268.71 | 1258.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 1258.60 | 1268.71 | 1258.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1265.80 | 1268.13 | 1259.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:30:00 | 1263.00 | 1268.13 | 1259.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1286.50 | 1271.46 | 1262.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 1292.40 | 1275.72 | 1271.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 1299.40 | 1287.34 | 1279.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:30:00 | 1291.40 | 1292.79 | 1286.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 1294.30 | 1292.54 | 1287.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1303.80 | 1294.79 | 1288.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:00:00 | 1309.50 | 1297.74 | 1290.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1314.90 | 1297.63 | 1292.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 10:15:00 | 1274.00 | 1292.75 | 1291.49 | SL hit (close<static) qty=1.00 sl=1287.00 alert=retest2 |

### Cycle 147 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1274.10 | 1289.02 | 1289.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 14:15:00 | 1261.70 | 1278.87 | 1284.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1268.00 | 1262.89 | 1271.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 1268.00 | 1262.89 | 1271.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1262.00 | 1262.72 | 1270.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1280.50 | 1262.72 | 1270.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1278.30 | 1265.83 | 1271.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 1284.20 | 1265.83 | 1271.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1268.60 | 1266.39 | 1271.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:00:00 | 1260.50 | 1265.21 | 1270.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 1275.50 | 1270.56 | 1269.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 1275.50 | 1270.56 | 1269.97 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 1260.60 | 1268.79 | 1269.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 14:15:00 | 1249.60 | 1261.73 | 1265.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 10:15:00 | 1238.10 | 1228.93 | 1236.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1238.10 | 1228.93 | 1236.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1238.10 | 1228.93 | 1236.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1238.10 | 1228.93 | 1236.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1243.30 | 1231.81 | 1236.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:30:00 | 1246.30 | 1231.81 | 1236.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1224.90 | 1232.73 | 1235.92 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1249.00 | 1237.59 | 1236.96 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 10:15:00 | 1231.10 | 1235.67 | 1236.16 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 11:15:00 | 1246.70 | 1237.88 | 1237.12 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 13:15:00 | 1232.20 | 1235.74 | 1236.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1227.90 | 1233.14 | 1234.83 | Break + close below crossover candle low |

### Cycle 154 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 1261.80 | 1237.80 | 1236.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 1270.10 | 1250.64 | 1243.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 1334.30 | 1339.58 | 1315.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 1334.30 | 1339.58 | 1315.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1319.20 | 1326.73 | 1317.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 1334.50 | 1326.73 | 1317.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 15:15:00 | 1334.00 | 1337.89 | 1338.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 15:15:00 | 1334.00 | 1337.89 | 1338.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1307.00 | 1331.72 | 1335.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 1323.50 | 1319.19 | 1326.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 15:00:00 | 1323.50 | 1319.19 | 1326.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1329.00 | 1321.15 | 1326.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1312.70 | 1321.15 | 1326.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 1331.30 | 1323.89 | 1327.22 | SL hit (close>static) qty=1.00 sl=1330.90 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 15:15:00 | 1330.90 | 1328.68 | 1328.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 1346.50 | 1332.24 | 1330.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1333.60 | 1348.35 | 1341.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1333.60 | 1348.35 | 1341.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1333.60 | 1348.35 | 1341.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1333.60 | 1348.35 | 1341.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1326.40 | 1343.96 | 1340.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1326.40 | 1343.96 | 1340.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1312.60 | 1337.69 | 1337.78 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1346.20 | 1336.88 | 1335.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1360.10 | 1345.64 | 1340.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 10:15:00 | 1372.50 | 1375.13 | 1360.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 11:00:00 | 1372.50 | 1375.13 | 1360.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1397.10 | 1409.03 | 1399.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1397.10 | 1409.03 | 1399.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1400.00 | 1407.23 | 1399.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1394.00 | 1407.23 | 1399.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1393.50 | 1404.48 | 1399.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:45:00 | 1398.90 | 1404.48 | 1399.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1391.10 | 1401.80 | 1398.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:45:00 | 1390.50 | 1401.80 | 1398.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1400.10 | 1397.43 | 1397.00 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1378.50 | 1393.96 | 1395.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1377.00 | 1390.57 | 1393.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 1387.70 | 1383.41 | 1387.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 1387.70 | 1383.41 | 1387.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1387.70 | 1383.41 | 1387.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 1387.90 | 1383.41 | 1387.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1397.20 | 1386.17 | 1388.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 1397.20 | 1386.17 | 1388.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1392.60 | 1387.45 | 1388.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 1398.20 | 1387.45 | 1388.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1387.40 | 1387.61 | 1388.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 1387.40 | 1387.61 | 1388.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1389.90 | 1388.07 | 1388.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 1394.90 | 1388.07 | 1388.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1386.80 | 1387.81 | 1388.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 1393.10 | 1387.81 | 1388.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 1395.10 | 1389.27 | 1389.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 1408.50 | 1393.12 | 1390.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 1395.00 | 1395.19 | 1392.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:00:00 | 1395.00 | 1395.19 | 1392.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1402.60 | 1396.67 | 1393.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1404.90 | 1396.67 | 1393.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1390.80 | 1397.21 | 1394.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:00:00 | 1390.80 | 1397.21 | 1394.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1383.60 | 1394.48 | 1393.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1383.30 | 1394.48 | 1393.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-07-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 13:15:00 | 1377.90 | 1391.17 | 1392.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 14:15:00 | 1374.90 | 1387.91 | 1390.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1362.10 | 1351.02 | 1363.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1362.10 | 1351.02 | 1363.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1362.10 | 1351.02 | 1363.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1365.00 | 1351.02 | 1363.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1372.60 | 1355.33 | 1364.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1380.10 | 1355.33 | 1364.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1370.30 | 1358.33 | 1364.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:45:00 | 1354.30 | 1359.53 | 1364.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 14:15:00 | 1372.20 | 1352.30 | 1351.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 14:15:00 | 1372.20 | 1352.30 | 1351.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 1382.50 | 1361.25 | 1355.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 15:15:00 | 1377.10 | 1377.95 | 1368.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:15:00 | 1395.60 | 1377.95 | 1368.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1384.70 | 1385.17 | 1374.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1384.70 | 1385.17 | 1374.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1385.00 | 1385.19 | 1378.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1369.40 | 1385.19 | 1378.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1363.60 | 1380.87 | 1376.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 1363.60 | 1380.87 | 1376.70 | SL hit (close<ema400) qty=1.00 sl=1376.70 alert=retest1 |

### Cycle 163 — SELL (started 2025-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 11:15:00 | 1356.20 | 1373.26 | 1373.80 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 1383.10 | 1374.65 | 1373.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 1397.80 | 1379.28 | 1376.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 09:15:00 | 1429.40 | 1430.17 | 1412.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:30:00 | 1430.10 | 1430.17 | 1412.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1423.10 | 1429.91 | 1421.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 1425.70 | 1429.91 | 1421.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1455.00 | 1437.99 | 1429.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 1467.30 | 1443.23 | 1433.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 1422.70 | 1433.89 | 1433.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 1422.70 | 1433.89 | 1433.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 12:15:00 | 1413.00 | 1429.71 | 1432.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 1409.00 | 1406.95 | 1415.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1380.60 | 1406.95 | 1415.88 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1387.00 | 1385.46 | 1396.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 1391.00 | 1385.46 | 1396.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 1389.90 | 1387.23 | 1395.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 1392.80 | 1387.23 | 1395.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1390.00 | 1388.56 | 1394.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 1390.00 | 1388.56 | 1394.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1385.50 | 1387.44 | 1392.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-30 10:15:00 | 1399.20 | 1389.79 | 1393.30 | SL hit (close>ema400) qty=1.00 sl=1393.30 alert=retest1 |

### Cycle 166 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1298.90 | 1260.22 | 1255.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 1323.90 | 1300.94 | 1291.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 15:15:00 | 1311.00 | 1311.90 | 1302.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:15:00 | 1303.80 | 1311.90 | 1302.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1300.00 | 1309.52 | 1302.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1300.00 | 1309.52 | 1302.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1297.00 | 1307.02 | 1301.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1297.30 | 1307.02 | 1301.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1298.00 | 1304.07 | 1301.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 1298.20 | 1304.07 | 1301.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1301.20 | 1303.65 | 1301.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 1301.20 | 1303.65 | 1301.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1299.70 | 1302.86 | 1301.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 1309.60 | 1302.86 | 1301.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1287.00 | 1301.58 | 1302.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1287.00 | 1301.58 | 1302.13 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 14:15:00 | 1302.20 | 1295.14 | 1295.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 10:15:00 | 1308.70 | 1298.20 | 1296.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 14:15:00 | 1300.50 | 1304.47 | 1300.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 14:15:00 | 1300.50 | 1304.47 | 1300.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 1300.50 | 1304.47 | 1300.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 1300.50 | 1304.47 | 1300.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1300.00 | 1303.58 | 1300.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 1287.90 | 1303.58 | 1300.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1295.70 | 1302.00 | 1300.10 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 11:15:00 | 1291.30 | 1298.65 | 1298.83 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1304.10 | 1299.74 | 1299.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 1314.30 | 1303.99 | 1301.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 1354.60 | 1356.21 | 1343.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 14:00:00 | 1354.60 | 1356.21 | 1343.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1350.90 | 1354.01 | 1344.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1357.60 | 1354.01 | 1344.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1373.00 | 1357.81 | 1347.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:45:00 | 1384.90 | 1371.39 | 1367.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1339.90 | 1361.02 | 1363.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 1339.90 | 1361.02 | 1363.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 1333.40 | 1355.49 | 1360.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 1354.00 | 1346.83 | 1354.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 1354.00 | 1346.83 | 1354.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1354.00 | 1346.83 | 1354.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 1354.00 | 1346.83 | 1354.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1350.00 | 1347.47 | 1353.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 1350.30 | 1347.47 | 1353.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1351.30 | 1348.23 | 1353.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:45:00 | 1355.60 | 1348.23 | 1353.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1350.50 | 1349.28 | 1352.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 1352.00 | 1349.28 | 1352.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1342.20 | 1347.86 | 1351.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:30:00 | 1337.60 | 1345.13 | 1350.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:00:00 | 1338.30 | 1342.19 | 1347.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:45:00 | 1338.20 | 1340.79 | 1346.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 10:15:00 | 1371.40 | 1351.15 | 1349.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 10:15:00 | 1371.40 | 1351.15 | 1349.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 11:15:00 | 1373.10 | 1355.54 | 1352.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 10:15:00 | 1360.60 | 1362.30 | 1357.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:30:00 | 1360.40 | 1362.30 | 1357.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1342.20 | 1358.28 | 1356.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:45:00 | 1343.80 | 1358.28 | 1356.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1346.70 | 1355.96 | 1355.42 | EMA400 retest candle locked (from upside) |

### Cycle 173 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1341.60 | 1353.09 | 1354.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 11:15:00 | 1332.40 | 1343.36 | 1348.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 10:15:00 | 1336.70 | 1336.36 | 1341.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:15:00 | 1337.20 | 1336.36 | 1341.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1334.80 | 1334.79 | 1339.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:45:00 | 1345.40 | 1334.79 | 1339.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1345.00 | 1336.83 | 1339.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1341.30 | 1336.83 | 1339.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1339.70 | 1337.41 | 1339.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:15:00 | 1330.40 | 1335.75 | 1338.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:30:00 | 1326.90 | 1331.29 | 1335.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 11:15:00 | 1263.88 | 1291.15 | 1306.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 12:15:00 | 1260.56 | 1283.92 | 1302.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-29 12:15:00 | 1197.36 | 1225.74 | 1249.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 174 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 1233.50 | 1192.75 | 1189.54 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1173.20 | 1192.04 | 1192.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1171.70 | 1187.97 | 1190.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1156.00 | 1155.62 | 1167.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 1156.00 | 1155.62 | 1167.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1161.00 | 1156.70 | 1167.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1180.90 | 1156.70 | 1167.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1169.00 | 1159.16 | 1167.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 14:00:00 | 1150.90 | 1160.39 | 1163.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1146.40 | 1157.97 | 1161.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 1179.60 | 1164.62 | 1163.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1179.60 | 1164.62 | 1163.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1183.40 | 1170.69 | 1166.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1334.30 | 1334.53 | 1283.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 1332.70 | 1334.53 | 1283.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1369.00 | 1383.17 | 1368.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 1369.00 | 1383.17 | 1368.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 1370.30 | 1380.59 | 1368.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 1365.20 | 1380.59 | 1368.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1367.50 | 1376.28 | 1368.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:30:00 | 1380.60 | 1376.61 | 1369.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:00:00 | 1384.70 | 1376.61 | 1369.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1340.30 | 1388.46 | 1394.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 1340.30 | 1388.46 | 1394.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1314.00 | 1348.91 | 1362.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 1338.60 | 1332.86 | 1349.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 13:30:00 | 1343.80 | 1332.86 | 1349.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1304.00 | 1329.35 | 1343.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:00:00 | 1297.00 | 1316.15 | 1333.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:00:00 | 1282.50 | 1304.73 | 1325.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 15:15:00 | 1232.15 | 1256.88 | 1285.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:15:00 | 1218.38 | 1229.37 | 1249.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 1225.10 | 1223.99 | 1241.34 | SL hit (close>ema200) qty=0.50 sl=1223.99 alert=retest2 |

### Cycle 178 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1195.80 | 1186.17 | 1186.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1200.00 | 1190.51 | 1188.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 1191.00 | 1193.06 | 1190.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 12:15:00 | 1191.00 | 1193.06 | 1190.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1191.00 | 1193.06 | 1190.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 1191.00 | 1193.06 | 1190.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1192.70 | 1192.99 | 1190.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 1188.00 | 1192.99 | 1190.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1200.10 | 1194.41 | 1191.31 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1081.10 | 1172.64 | 1182.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 1026.30 | 1067.27 | 1096.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 1011.20 | 1008.33 | 1030.97 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:15:00 | 994.20 | 1008.33 | 1030.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 09:15:00 | 944.49 | 955.04 | 972.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-12-09 09:15:00 | 894.78 | 921.44 | 944.47 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 180 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 926.30 | 924.86 | 924.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 933.70 | 926.77 | 925.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 937.30 | 941.94 | 938.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 937.30 | 941.94 | 938.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 937.30 | 941.94 | 938.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 937.30 | 941.94 | 938.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 938.60 | 941.27 | 938.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:30:00 | 932.10 | 941.27 | 938.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 933.20 | 939.65 | 937.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:00:00 | 933.20 | 939.65 | 937.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 933.50 | 938.42 | 937.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 933.10 | 938.42 | 937.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 929.80 | 936.70 | 936.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 924.30 | 934.22 | 935.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 926.10 | 925.64 | 929.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:30:00 | 926.00 | 925.64 | 929.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 918.00 | 924.32 | 928.14 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 938.80 | 928.18 | 928.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 953.10 | 933.16 | 930.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 946.00 | 947.52 | 940.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:00:00 | 946.00 | 947.52 | 940.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 939.70 | 945.25 | 940.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 939.70 | 945.25 | 940.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 941.50 | 944.50 | 940.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 14:45:00 | 944.00 | 943.74 | 940.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 934.40 | 941.74 | 940.38 | SL hit (close<static) qty=1.00 sl=937.90 alert=retest2 |

### Cycle 183 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 925.70 | 936.83 | 938.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 923.90 | 934.24 | 936.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 913.90 | 912.79 | 919.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:45:00 | 915.70 | 912.79 | 919.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 907.20 | 898.39 | 903.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:00:00 | 907.20 | 898.39 | 903.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 900.50 | 898.81 | 903.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 905.70 | 898.81 | 903.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 899.90 | 899.03 | 903.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 901.80 | 899.03 | 903.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 896.90 | 898.74 | 902.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 10:30:00 | 894.00 | 897.75 | 901.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 13:15:00 | 902.65 | 898.62 | 898.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 902.65 | 898.62 | 898.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 906.45 | 901.63 | 900.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 906.00 | 910.64 | 906.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 906.00 | 910.64 | 906.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 906.00 | 910.64 | 906.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 900.95 | 910.64 | 906.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 898.80 | 908.27 | 905.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 898.80 | 908.27 | 905.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 890.00 | 904.62 | 904.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 890.00 | 904.62 | 904.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 890.45 | 901.79 | 903.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 887.30 | 898.89 | 901.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 806.95 | 806.19 | 816.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-16 10:00:00 | 806.95 | 806.19 | 816.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 798.10 | 788.55 | 793.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:00:00 | 798.10 | 788.55 | 793.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 796.00 | 790.04 | 793.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 15:00:00 | 786.85 | 790.19 | 793.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 803.00 | 792.61 | 793.94 | SL hit (close>static) qty=1.00 sl=801.80 alert=retest2 |

### Cycle 186 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 797.95 | 793.96 | 793.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 11:15:00 | 804.75 | 796.78 | 795.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 12:15:00 | 790.70 | 795.57 | 794.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 12:15:00 | 790.70 | 795.57 | 794.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 790.70 | 795.57 | 794.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 13:00:00 | 790.70 | 795.57 | 794.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 783.10 | 793.07 | 793.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 778.00 | 788.28 | 791.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 15:15:00 | 785.00 | 783.99 | 787.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 15:15:00 | 785.00 | 783.99 | 787.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 785.00 | 783.99 | 787.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 778.15 | 783.99 | 787.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 09:15:00 | 796.20 | 781.24 | 780.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 796.20 | 781.24 | 780.82 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 771.05 | 780.51 | 781.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 765.60 | 775.84 | 779.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 775.00 | 774.83 | 777.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 775.00 | 774.83 | 777.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 775.00 | 774.83 | 777.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:30:00 | 779.00 | 774.83 | 777.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 776.90 | 775.24 | 777.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 776.90 | 775.24 | 777.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 782.15 | 776.62 | 778.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 782.15 | 776.62 | 778.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 792.65 | 779.83 | 779.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 797.75 | 783.41 | 781.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 15:15:00 | 789.25 | 790.09 | 786.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:15:00 | 790.30 | 790.09 | 786.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 802.50 | 792.57 | 787.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 822.00 | 792.75 | 789.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 14:15:00 | 904.20 | 883.18 | 856.68 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 191 — SELL (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 10:15:00 | 884.70 | 894.04 | 894.63 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 902.00 | 895.16 | 894.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 910.40 | 898.21 | 896.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 920.50 | 927.69 | 920.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 920.50 | 927.69 | 920.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 920.50 | 927.69 | 920.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:00:00 | 920.50 | 927.69 | 920.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 918.10 | 925.77 | 920.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 920.40 | 925.77 | 920.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 912.10 | 923.03 | 919.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 912.10 | 923.03 | 919.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 915.00 | 917.80 | 917.90 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 930.50 | 919.92 | 918.81 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 917.00 | 922.54 | 923.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 13:15:00 | 901.95 | 918.09 | 920.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 920.65 | 913.91 | 917.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 09:15:00 | 920.65 | 913.91 | 917.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 920.65 | 913.91 | 917.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 920.65 | 913.91 | 917.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 920.05 | 915.14 | 918.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 915.00 | 916.78 | 918.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:00:00 | 917.90 | 917.61 | 918.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 14:15:00 | 916.95 | 914.09 | 914.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 920.80 | 915.43 | 914.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — BUY (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 14:15:00 | 920.80 | 915.43 | 914.73 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 900.45 | 913.18 | 913.87 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 915.30 | 906.49 | 905.37 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 14:15:00 | 894.80 | 903.95 | 904.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 866.60 | 895.05 | 900.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 11:15:00 | 876.50 | 874.81 | 883.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:30:00 | 878.85 | 874.81 | 883.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 880.00 | 875.51 | 880.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 881.50 | 875.51 | 880.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 895.80 | 879.57 | 882.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 895.80 | 879.57 | 882.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 899.85 | 883.62 | 883.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 899.85 | 883.62 | 883.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 200 — BUY (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 11:15:00 | 897.70 | 886.44 | 885.03 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 874.75 | 884.41 | 885.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 863.40 | 875.34 | 880.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 823.35 | 822.95 | 836.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 09:30:00 | 822.50 | 822.95 | 836.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 830.85 | 823.59 | 829.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 836.90 | 823.59 | 829.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 835.95 | 826.06 | 830.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 838.65 | 826.06 | 830.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 833.55 | 827.56 | 830.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:45:00 | 835.50 | 827.56 | 830.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 839.50 | 829.95 | 831.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 839.50 | 829.95 | 831.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 845.40 | 833.04 | 832.80 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 822.50 | 832.00 | 832.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 814.40 | 826.26 | 829.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 831.00 | 823.66 | 827.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 831.00 | 823.66 | 827.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 831.00 | 823.66 | 827.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 831.00 | 823.66 | 827.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 824.00 | 823.73 | 827.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:30:00 | 819.15 | 825.72 | 826.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 793.70 | 827.47 | 827.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:15:00 | 778.19 | 802.07 | 813.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 793.00 | 788.29 | 799.34 | SL hit (close>ema200) qty=0.50 sl=788.29 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 809.40 | 803.05 | 802.21 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 794.65 | 801.05 | 801.56 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 813.45 | 802.69 | 802.07 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 778.45 | 797.99 | 800.06 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 812.40 | 799.47 | 798.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 824.65 | 804.51 | 800.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 805.00 | 816.00 | 809.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 805.00 | 816.00 | 809.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 805.00 | 816.00 | 809.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 805.00 | 816.00 | 809.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 801.10 | 813.02 | 808.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:45:00 | 805.10 | 811.30 | 808.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:45:00 | 808.60 | 810.43 | 807.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:45:00 | 806.65 | 807.70 | 807.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 805.30 | 807.70 | 807.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 11:15:00 | 806.30 | 807.42 | 807.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 11:15:00 | 806.30 | 807.42 | 807.54 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 815.15 | 808.97 | 808.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 822.45 | 811.66 | 809.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 812.75 | 814.18 | 811.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 812.75 | 814.18 | 811.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 812.75 | 814.18 | 811.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 809.50 | 814.18 | 811.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 808.80 | 813.10 | 811.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:00:00 | 808.80 | 813.10 | 811.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 818.40 | 814.16 | 811.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 820.40 | 815.67 | 812.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 820.40 | 829.32 | 829.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 820.40 | 829.32 | 829.75 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 852.75 | 829.45 | 828.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 12:15:00 | 874.00 | 855.47 | 847.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 916.75 | 917.74 | 898.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 11:45:00 | 918.20 | 917.74 | 898.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 984.00 | 992.05 | 981.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 985.45 | 992.05 | 981.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 989.15 | 991.47 | 982.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:45:00 | 979.85 | 991.47 | 982.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 981.35 | 990.28 | 986.06 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 12:15:00 | 972.75 | 983.20 | 983.55 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 998.40 | 986.24 | 984.90 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 978.25 | 987.25 | 987.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 11:15:00 | 971.00 | 977.96 | 981.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 10:15:00 | 980.00 | 973.51 | 976.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 10:15:00 | 980.00 | 973.51 | 976.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 980.00 | 973.51 | 976.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 980.00 | 973.51 | 976.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 979.10 | 974.63 | 977.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 972.00 | 976.39 | 977.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-12 09:15:00 | 1426.55 | 2024-04-23 09:15:00 | 1569.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-12 15:15:00 | 1426.10 | 2024-04-23 09:15:00 | 1568.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-15 10:30:00 | 1425.00 | 2024-04-23 09:15:00 | 1567.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-15 13:00:00 | 1422.15 | 2024-04-23 09:15:00 | 1564.37 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-19 15:00:00 | 1440.00 | 2024-04-23 09:15:00 | 1584.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-02 10:00:00 | 1489.50 | 2024-05-07 11:15:00 | 1415.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 15:15:00 | 1486.00 | 2024-05-07 11:15:00 | 1411.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 09:30:00 | 1488.75 | 2024-05-07 11:15:00 | 1414.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-02 10:00:00 | 1489.50 | 2024-05-07 13:15:00 | 1438.45 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2024-05-02 15:15:00 | 1486.00 | 2024-05-07 13:15:00 | 1438.45 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2024-05-03 09:30:00 | 1488.75 | 2024-05-07 13:15:00 | 1438.45 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2024-05-27 11:45:00 | 1524.30 | 2024-05-28 10:15:00 | 1538.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-05-27 13:15:00 | 1524.05 | 2024-05-28 10:15:00 | 1538.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-05-28 10:00:00 | 1525.05 | 2024-05-28 10:15:00 | 1538.90 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-06-03 10:15:00 | 1492.90 | 2024-06-03 14:15:00 | 1514.45 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-06-03 12:00:00 | 1494.70 | 2024-06-03 14:15:00 | 1514.45 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-18 11:30:00 | 1794.60 | 2024-06-26 12:15:00 | 1833.00 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2024-07-02 09:15:00 | 1965.00 | 2024-07-02 09:15:00 | 1934.85 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-07-09 12:15:00 | 2006.90 | 2024-07-11 15:15:00 | 2000.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-07-18 12:15:00 | 2043.70 | 2024-07-19 12:15:00 | 2015.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-07-18 14:30:00 | 2039.05 | 2024-07-19 12:15:00 | 2015.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-01 13:15:00 | 2131.65 | 2024-08-05 09:15:00 | 2070.00 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-08-01 14:30:00 | 2142.55 | 2024-08-05 09:15:00 | 2070.00 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-08-01 15:15:00 | 2148.00 | 2024-08-05 09:15:00 | 2070.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-08-02 11:45:00 | 2130.60 | 2024-08-05 09:15:00 | 2070.00 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2024-08-21 13:15:00 | 2072.30 | 2024-08-28 11:15:00 | 2047.95 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2024-08-22 09:30:00 | 2067.80 | 2024-08-28 11:15:00 | 2047.95 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2024-08-22 10:30:00 | 2071.65 | 2024-08-28 11:15:00 | 2047.95 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2024-09-04 11:30:00 | 2200.65 | 2024-09-09 09:15:00 | 2178.15 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-04 14:15:00 | 2222.80 | 2024-09-09 09:15:00 | 2178.15 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-09-05 13:15:00 | 2208.55 | 2024-09-09 09:15:00 | 2178.15 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-09-16 09:30:00 | 2090.05 | 2024-09-17 09:15:00 | 1985.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-16 09:30:00 | 2090.05 | 2024-09-18 09:15:00 | 2070.85 | STOP_HIT | 0.50 | 0.92% |
| BUY | retest2 | 2024-09-25 15:00:00 | 2119.90 | 2024-09-30 11:15:00 | 2331.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-07 12:30:00 | 2079.55 | 2024-11-08 13:15:00 | 2039.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2024-11-14 12:45:00 | 1837.30 | 2024-11-19 14:15:00 | 1835.00 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-11-14 13:45:00 | 1834.15 | 2024-11-19 14:15:00 | 1835.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-11-19 11:30:00 | 1822.00 | 2024-11-19 14:15:00 | 1835.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2024-11-19 13:30:00 | 1834.00 | 2024-11-19 14:15:00 | 1835.00 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-11-27 12:15:00 | 1799.85 | 2024-12-09 14:15:00 | 1884.90 | STOP_HIT | 1.00 | 4.73% |
| BUY | retest2 | 2024-11-28 09:15:00 | 1811.25 | 2024-12-09 14:15:00 | 1884.90 | STOP_HIT | 1.00 | 4.07% |
| BUY | retest2 | 2024-12-11 12:30:00 | 1912.75 | 2024-12-13 15:15:00 | 1904.80 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-12-13 11:15:00 | 1911.00 | 2024-12-13 15:15:00 | 1904.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-12-19 13:45:00 | 1955.30 | 2024-12-20 15:15:00 | 1915.10 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-12-19 14:45:00 | 1961.15 | 2024-12-20 15:15:00 | 1915.10 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-12-19 15:15:00 | 1962.15 | 2024-12-20 15:15:00 | 1915.10 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-12-20 09:30:00 | 1958.25 | 2024-12-20 15:15:00 | 1915.10 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1870.05 | 2024-12-30 14:15:00 | 1781.01 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2024-12-27 09:30:00 | 1874.75 | 2024-12-30 14:15:00 | 1783.77 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2024-12-27 14:15:00 | 1877.65 | 2024-12-30 14:15:00 | 1778.31 | PARTIAL | 0.50 | 5.29% |
| SELL | retest2 | 2024-12-26 09:15:00 | 1870.05 | 2024-12-31 14:15:00 | 1836.70 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2024-12-27 09:30:00 | 1874.75 | 2024-12-31 14:15:00 | 1836.70 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2024-12-27 14:15:00 | 1877.65 | 2024-12-31 14:15:00 | 1836.70 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2024-12-27 15:00:00 | 1871.90 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | 1.24% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1850.00 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2024-12-30 09:45:00 | 1860.95 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2024-12-30 12:00:00 | 1859.75 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2024-12-30 13:15:00 | 1838.20 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-12-30 14:30:00 | 1790.25 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | -3.26% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1795.50 | 2025-01-02 14:15:00 | 1848.70 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1674.90 | 2025-01-13 14:15:00 | 1591.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-13 09:15:00 | 1674.90 | 2025-01-14 11:15:00 | 1640.05 | STOP_HIT | 0.50 | 2.08% |
| SELL | retest2 | 2025-01-15 09:15:00 | 1694.80 | 2025-01-15 10:15:00 | 1695.80 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-01-15 09:45:00 | 1697.45 | 2025-01-15 10:15:00 | 1695.80 | STOP_HIT | 1.00 | 0.10% |
| SELL | retest2 | 2025-01-20 09:15:00 | 1670.95 | 2025-01-22 09:15:00 | 1587.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 12:45:00 | 1666.70 | 2025-01-22 09:15:00 | 1583.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 1670.95 | 2025-01-23 09:15:00 | 1600.85 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-01-20 12:45:00 | 1666.70 | 2025-01-23 09:15:00 | 1600.85 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2025-02-27 10:45:00 | 954.30 | 2025-03-03 09:15:00 | 906.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 10:45:00 | 954.30 | 2025-03-03 10:15:00 | 927.65 | STOP_HIT | 0.50 | 2.79% |
| SELL | retest2 | 2025-02-28 09:15:00 | 945.00 | 2025-03-04 15:15:00 | 942.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2025-03-07 09:15:00 | 999.65 | 2025-03-10 10:15:00 | 978.10 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-03-13 15:15:00 | 939.80 | 2025-03-18 09:15:00 | 968.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-03-17 14:00:00 | 939.85 | 2025-03-18 09:15:00 | 968.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-03-17 15:00:00 | 934.00 | 2025-03-18 09:15:00 | 968.00 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2025-03-24 10:15:00 | 1010.00 | 2025-03-25 15:15:00 | 984.10 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-04-01 12:00:00 | 1019.20 | 2025-04-07 09:15:00 | 986.50 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-04-11 09:15:00 | 1077.15 | 2025-04-21 11:15:00 | 1094.00 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2025-05-16 12:15:00 | 1292.40 | 2025-05-21 10:15:00 | 1274.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-05-19 09:15:00 | 1299.40 | 2025-05-21 10:15:00 | 1274.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-05-19 14:30:00 | 1291.40 | 2025-05-21 11:15:00 | 1274.10 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-05-20 10:15:00 | 1294.30 | 2025-05-21 11:15:00 | 1274.10 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-05-20 12:00:00 | 1309.50 | 2025-05-21 11:15:00 | 1274.10 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1314.90 | 2025-05-21 11:15:00 | 1274.10 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-05-23 12:00:00 | 1260.50 | 2025-05-26 13:15:00 | 1275.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-06-11 09:15:00 | 1334.50 | 2025-06-13 15:15:00 | 1334.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1312.70 | 2025-06-17 10:15:00 | 1331.30 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-06-17 10:30:00 | 1321.20 | 2025-06-17 11:15:00 | 1331.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-09 13:45:00 | 1354.30 | 2025-07-11 14:15:00 | 1372.20 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2025-07-15 09:15:00 | 1395.60 | 2025-07-16 09:15:00 | 1363.60 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-23 10:30:00 | 1467.30 | 2025-07-24 11:15:00 | 1422.70 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest1 | 2025-07-28 09:15:00 | 1380.60 | 2025-07-30 10:15:00 | 1399.20 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1362.80 | 2025-08-07 09:15:00 | 1294.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1362.80 | 2025-08-07 09:15:00 | 1322.00 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest2 | 2025-08-25 09:15:00 | 1309.60 | 2025-08-26 09:15:00 | 1287.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-09-11 09:45:00 | 1384.90 | 2025-09-11 12:15:00 | 1339.90 | STOP_HIT | 1.00 | -3.25% |
| SELL | retest2 | 2025-09-15 10:30:00 | 1337.60 | 2025-09-16 10:15:00 | 1371.40 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-09-15 13:00:00 | 1338.30 | 2025-09-16 10:15:00 | 1371.40 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-09-15 13:45:00 | 1338.20 | 2025-09-16 10:15:00 | 1371.40 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-09-22 13:15:00 | 1330.40 | 2025-09-25 11:15:00 | 1263.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:30:00 | 1326.90 | 2025-09-25 12:15:00 | 1260.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 13:15:00 | 1330.40 | 2025-09-29 12:15:00 | 1197.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-23 09:30:00 | 1326.90 | 2025-09-29 12:15:00 | 1194.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-14 14:00:00 | 1150.90 | 2025-10-15 12:15:00 | 1179.60 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-15 09:15:00 | 1146.40 | 2025-10-15 12:15:00 | 1179.60 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-10-27 11:30:00 | 1380.60 | 2025-11-03 09:15:00 | 1340.30 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2025-10-27 12:00:00 | 1384.70 | 2025-11-03 09:15:00 | 1340.30 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-11-10 13:00:00 | 1297.00 | 2025-11-11 15:15:00 | 1232.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1282.50 | 2025-11-13 11:15:00 | 1218.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-10 13:00:00 | 1297.00 | 2025-11-13 14:15:00 | 1225.10 | STOP_HIT | 0.50 | 5.54% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1282.50 | 2025-11-13 14:15:00 | 1225.10 | STOP_HIT | 0.50 | 4.48% |
| SELL | retest1 | 2025-12-03 09:15:00 | 994.20 | 2025-12-08 09:15:00 | 944.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-12-03 09:15:00 | 994.20 | 2025-12-09 09:15:00 | 894.78 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-12-23 14:45:00 | 944.00 | 2025-12-24 09:15:00 | 934.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-01 10:30:00 | 894.00 | 2026-01-02 13:15:00 | 902.65 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-21 15:00:00 | 786.85 | 2026-01-22 09:15:00 | 803.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-01-22 10:30:00 | 790.25 | 2026-01-23 11:15:00 | 797.95 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-22 14:30:00 | 791.05 | 2026-01-23 11:15:00 | 797.95 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-01-29 09:15:00 | 778.15 | 2026-02-01 09:15:00 | 796.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-02-05 09:15:00 | 822.00 | 2026-02-09 14:15:00 | 904.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 915.00 | 2026-02-27 14:15:00 | 920.80 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-02-26 10:00:00 | 917.90 | 2026-02-27 14:15:00 | 920.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-02-27 14:15:00 | 916.95 | 2026-02-27 14:15:00 | 920.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-03-20 14:30:00 | 819.15 | 2026-03-23 12:15:00 | 778.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 14:30:00 | 819.15 | 2026-03-24 12:15:00 | 793.00 | STOP_HIT | 0.50 | 3.19% |
| SELL | retest2 | 2026-03-23 09:15:00 | 793.70 | 2026-03-25 14:15:00 | 809.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-04-02 11:45:00 | 805.10 | 2026-04-06 11:15:00 | 806.30 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2026-04-02 12:45:00 | 808.60 | 2026-04-06 11:15:00 | 806.30 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2026-04-06 10:45:00 | 806.65 | 2026-04-06 11:15:00 | 806.30 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-04-06 11:15:00 | 805.30 | 2026-04-06 11:15:00 | 806.30 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2026-04-07 12:30:00 | 820.40 | 2026-04-13 09:15:00 | 820.40 | STOP_HIT | 1.00 | 0.00% |
