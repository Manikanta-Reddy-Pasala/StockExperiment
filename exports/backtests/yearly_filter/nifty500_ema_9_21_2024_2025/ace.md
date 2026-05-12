# Action Construction Equipment Ltd. (ACE)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 949.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 137 |
| ALERT1 | 99 |
| ALERT2 | 96 |
| ALERT2_SKIP | 50 |
| ALERT3 | 256 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 114 |
| PARTIAL | 24 |
| TARGET_HIT | 12 |
| STOP_HIT | 104 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 140 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 66 / 74
- **Target hits / Stop hits / Partials:** 12 / 104 / 24
- **Avg / median % per leg:** 1.31% / -0.22%
- **Sum % (uncompounded):** 183.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 13 | 28.9% | 7 | 38 | 0 | 0.76% | 34.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.93% | -1.9% |
| BUY @ 3rd Alert (retest2) | 44 | 13 | 29.5% | 7 | 37 | 0 | 0.82% | 36.2% |
| SELL (all) | 95 | 53 | 55.8% | 5 | 66 | 24 | 1.57% | 149.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| SELL @ 3rd Alert (retest2) | 94 | 53 | 56.4% | 5 | 65 | 24 | 1.60% | 150.6% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.50% | -3.0% |
| retest2 (combined) | 138 | 66 | 47.8% | 12 | 102 | 24 | 1.35% | 186.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 1382.00 | 1372.89 | 1372.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 1399.10 | 1381.82 | 1376.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 11:15:00 | 1390.35 | 1391.99 | 1384.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 1390.35 | 1391.99 | 1384.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 1390.35 | 1391.99 | 1384.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 11:45:00 | 1382.90 | 1391.99 | 1384.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 1393.50 | 1392.30 | 1385.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1434.30 | 1393.40 | 1387.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-21 14:15:00 | 1577.73 | 1526.59 | 1478.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 1494.50 | 1508.93 | 1510.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 12:15:00 | 1476.45 | 1502.44 | 1507.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 1396.60 | 1396.43 | 1423.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 14:15:00 | 1420.20 | 1403.64 | 1416.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1420.20 | 1403.64 | 1416.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 15:15:00 | 1429.40 | 1403.64 | 1416.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1429.40 | 1408.79 | 1417.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1433.00 | 1408.79 | 1417.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1431.50 | 1413.32 | 1418.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:30:00 | 1433.05 | 1413.32 | 1418.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 1432.30 | 1417.11 | 1419.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 12:00:00 | 1432.30 | 1417.11 | 1419.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 14:15:00 | 1424.00 | 1421.35 | 1421.17 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 09:15:00 | 1404.00 | 1419.05 | 1420.23 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 1435.90 | 1421.80 | 1421.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1457.20 | 1429.54 | 1424.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1405.35 | 1437.01 | 1433.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1405.35 | 1437.01 | 1433.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1405.35 | 1437.01 | 1433.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1401.00 | 1437.01 | 1433.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1354.75 | 1420.56 | 1426.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1290.50 | 1394.55 | 1413.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1367.50 | 1352.81 | 1379.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1367.50 | 1352.81 | 1379.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1399.10 | 1362.07 | 1381.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 14:00:00 | 1350.00 | 1363.70 | 1379.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 12:15:00 | 1403.95 | 1383.19 | 1382.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 1403.95 | 1383.19 | 1382.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1429.80 | 1399.62 | 1390.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 1470.10 | 1477.53 | 1453.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 15:15:00 | 1470.10 | 1477.53 | 1453.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1470.10 | 1477.53 | 1453.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1486.45 | 1477.53 | 1453.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 09:15:00 | 1430.00 | 1468.03 | 1451.55 | SL hit (close<static) qty=1.00 sl=1450.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 15:15:00 | 1435.00 | 1442.17 | 1443.15 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 12:15:00 | 1448.55 | 1444.53 | 1444.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 1456.45 | 1447.95 | 1445.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 1447.05 | 1447.77 | 1445.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 10:15:00 | 1447.05 | 1447.77 | 1445.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1447.05 | 1447.77 | 1445.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 1447.05 | 1447.77 | 1445.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1453.35 | 1448.89 | 1446.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:15:00 | 1455.25 | 1449.51 | 1447.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 15:15:00 | 1456.00 | 1451.90 | 1448.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 09:30:00 | 1462.20 | 1474.92 | 1470.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:00:00 | 1457.10 | 1471.35 | 1468.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 12:15:00 | 1453.85 | 1465.24 | 1466.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 12:15:00 | 1453.85 | 1465.24 | 1466.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 13:15:00 | 1448.75 | 1461.95 | 1464.85 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 14:15:00 | 1510.55 | 1471.67 | 1469.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 10:15:00 | 1525.15 | 1496.14 | 1482.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 09:15:00 | 1494.20 | 1507.66 | 1495.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 1494.20 | 1507.66 | 1495.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1494.20 | 1507.66 | 1495.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:00:00 | 1494.20 | 1507.66 | 1495.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 1491.30 | 1504.39 | 1495.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 1491.35 | 1504.39 | 1495.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 1494.45 | 1502.40 | 1495.30 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 10:15:00 | 1476.75 | 1490.86 | 1491.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 1471.05 | 1482.93 | 1487.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 1484.35 | 1481.44 | 1485.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1484.35 | 1481.44 | 1485.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1484.35 | 1481.44 | 1485.62 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 1509.05 | 1491.38 | 1489.12 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 1490.30 | 1498.36 | 1498.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 15:15:00 | 1484.00 | 1495.49 | 1497.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 09:15:00 | 1486.60 | 1472.61 | 1480.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 1486.60 | 1472.61 | 1480.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1486.60 | 1472.61 | 1480.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:15:00 | 1490.50 | 1472.61 | 1480.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1497.00 | 1477.49 | 1482.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:00:00 | 1497.00 | 1477.49 | 1482.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1488.15 | 1479.62 | 1482.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-02 12:45:00 | 1473.45 | 1479.86 | 1482.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 15:15:00 | 1493.00 | 1484.42 | 1484.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 15:15:00 | 1493.00 | 1484.42 | 1484.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1553.85 | 1498.31 | 1490.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 1526.00 | 1526.65 | 1510.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 15:00:00 | 1526.00 | 1526.65 | 1510.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1498.25 | 1521.97 | 1517.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:30:00 | 1495.70 | 1521.97 | 1517.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 1491.95 | 1515.97 | 1515.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:00:00 | 1491.95 | 1515.97 | 1515.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 1497.70 | 1512.31 | 1513.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 14:15:00 | 1491.00 | 1503.15 | 1508.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 14:15:00 | 1466.90 | 1458.50 | 1471.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:00:00 | 1466.90 | 1458.50 | 1471.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 15:15:00 | 1478.00 | 1462.40 | 1472.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 09:15:00 | 1467.95 | 1462.40 | 1472.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1447.00 | 1459.32 | 1469.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 1441.25 | 1459.32 | 1469.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:15:00 | 1440.00 | 1456.67 | 1467.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 09:45:00 | 1445.00 | 1435.79 | 1450.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 1435.00 | 1437.02 | 1448.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1424.90 | 1412.75 | 1423.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-15 15:15:00 | 1447.30 | 1429.56 | 1427.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 15:15:00 | 1447.30 | 1429.56 | 1427.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1454.65 | 1434.57 | 1430.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 1438.25 | 1440.98 | 1435.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 1438.25 | 1440.98 | 1435.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1438.25 | 1440.98 | 1435.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1438.25 | 1440.98 | 1435.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1420.10 | 1437.71 | 1435.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1423.25 | 1437.71 | 1435.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1415.65 | 1433.30 | 1433.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1369.85 | 1413.19 | 1423.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1394.75 | 1384.72 | 1400.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 1394.75 | 1384.72 | 1400.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1385.00 | 1384.78 | 1398.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1396.25 | 1384.78 | 1398.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1398.00 | 1387.35 | 1396.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 1396.80 | 1387.35 | 1396.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 1407.00 | 1391.28 | 1397.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 1407.00 | 1391.28 | 1397.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1412.80 | 1395.59 | 1398.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1412.00 | 1395.59 | 1398.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1411.40 | 1399.44 | 1400.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 1411.40 | 1399.44 | 1400.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1397.90 | 1399.13 | 1399.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1355.00 | 1399.13 | 1399.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 1379.30 | 1395.17 | 1397.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 1377.75 | 1394.13 | 1397.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 1413.10 | 1396.68 | 1397.47 | SL hit (close>static) qty=1.00 sl=1411.40 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1418.95 | 1401.14 | 1399.42 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 12:15:00 | 1393.90 | 1402.18 | 1402.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 15:15:00 | 1389.95 | 1394.25 | 1397.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 1395.00 | 1394.40 | 1397.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 1395.00 | 1394.40 | 1397.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1395.00 | 1394.40 | 1397.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 14:00:00 | 1389.25 | 1393.94 | 1396.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 10:15:00 | 1441.10 | 1402.57 | 1399.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 1441.10 | 1402.57 | 1399.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 10:15:00 | 1453.10 | 1433.13 | 1418.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 1426.95 | 1438.02 | 1425.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 13:15:00 | 1426.95 | 1438.02 | 1425.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 1426.95 | 1438.02 | 1425.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:45:00 | 1426.00 | 1438.02 | 1425.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1390.55 | 1428.53 | 1422.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1390.55 | 1428.53 | 1422.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1389.40 | 1420.70 | 1419.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 1368.40 | 1420.70 | 1419.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1363.60 | 1409.28 | 1414.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 14:15:00 | 1360.95 | 1379.26 | 1395.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1326.95 | 1306.35 | 1330.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1326.95 | 1306.35 | 1330.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1326.95 | 1306.35 | 1330.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:00:00 | 1293.05 | 1304.84 | 1325.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 1297.60 | 1290.91 | 1304.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:15:00 | 1298.95 | 1304.60 | 1306.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 14:45:00 | 1299.30 | 1303.96 | 1305.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1310.40 | 1305.25 | 1306.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 1318.00 | 1305.25 | 1306.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1305.55 | 1305.31 | 1306.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1301.05 | 1305.31 | 1306.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1297.75 | 1303.80 | 1305.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:45:00 | 1294.50 | 1295.24 | 1299.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:00:00 | 1299.75 | 1298.50 | 1300.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 1304.55 | 1299.71 | 1300.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:30:00 | 1302.00 | 1299.71 | 1300.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1300.05 | 1299.78 | 1300.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 14:15:00 | 1299.90 | 1299.78 | 1300.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 1287.95 | 1299.08 | 1300.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1232.72 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1234.00 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1234.33 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1236.00 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1232.86 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1234.76 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 14:15:00 | 1234.90 | 1254.05 | 1269.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 1228.40 | 1251.23 | 1266.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-14 15:15:00 | 1229.77 | 1251.23 | 1266.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-16 09:15:00 | 1261.95 | 1253.37 | 1266.49 | SL hit (close>ema200) qty=0.50 sl=1253.37 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1299.50 | 1268.59 | 1268.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 1315.85 | 1290.94 | 1280.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1308.60 | 1311.81 | 1297.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 1308.60 | 1311.81 | 1297.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1312.35 | 1311.53 | 1300.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 14:30:00 | 1318.60 | 1315.47 | 1306.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:45:00 | 1318.30 | 1322.47 | 1314.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 15:00:00 | 1321.30 | 1322.24 | 1315.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 14:15:00 | 1306.70 | 1312.99 | 1313.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 1306.70 | 1312.99 | 1313.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 1296.95 | 1309.31 | 1311.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 1302.85 | 1302.12 | 1305.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 1302.85 | 1302.12 | 1305.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 1302.85 | 1302.12 | 1305.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 1298.75 | 1303.67 | 1305.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 14:00:00 | 1296.00 | 1302.14 | 1304.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 1288.90 | 1304.09 | 1305.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 15:15:00 | 1294.60 | 1282.97 | 1281.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1294.60 | 1282.97 | 1281.82 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 1272.50 | 1279.86 | 1280.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 1266.40 | 1277.17 | 1279.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 1278.45 | 1274.21 | 1276.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 1278.45 | 1274.21 | 1276.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1278.45 | 1274.21 | 1276.60 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 13:15:00 | 1280.85 | 1278.06 | 1277.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 1294.85 | 1281.42 | 1279.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 1284.10 | 1286.69 | 1283.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:00:00 | 1284.10 | 1286.69 | 1283.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1281.00 | 1285.55 | 1282.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 1281.00 | 1285.55 | 1282.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1281.10 | 1284.66 | 1282.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:30:00 | 1281.00 | 1284.66 | 1282.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 1279.65 | 1283.66 | 1282.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 1279.65 | 1283.66 | 1282.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 1279.15 | 1282.76 | 1282.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 1272.95 | 1282.76 | 1282.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 1269.35 | 1280.08 | 1280.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 12:15:00 | 1263.25 | 1269.40 | 1273.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 1249.90 | 1243.83 | 1253.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 1249.90 | 1243.83 | 1253.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1249.90 | 1243.83 | 1253.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:45:00 | 1246.15 | 1245.16 | 1252.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 13:30:00 | 1245.95 | 1246.52 | 1251.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 10:15:00 | 1260.65 | 1254.84 | 1254.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 10:15:00 | 1260.65 | 1254.84 | 1254.57 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 13:15:00 | 1249.30 | 1253.62 | 1254.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 09:15:00 | 1247.50 | 1251.85 | 1253.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 15:15:00 | 1249.75 | 1249.00 | 1250.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-13 09:15:00 | 1256.50 | 1249.00 | 1250.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1251.45 | 1249.49 | 1250.94 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 1275.80 | 1255.98 | 1253.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 11:15:00 | 1289.45 | 1262.67 | 1256.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 13:15:00 | 1357.50 | 1364.42 | 1338.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:45:00 | 1359.25 | 1364.42 | 1338.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1328.05 | 1352.79 | 1339.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 1332.00 | 1352.79 | 1339.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1310.45 | 1344.32 | 1336.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 1310.45 | 1344.32 | 1336.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 13:15:00 | 1314.30 | 1329.44 | 1330.93 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1367.80 | 1337.62 | 1334.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1416.30 | 1364.50 | 1348.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 1434.00 | 1439.12 | 1421.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 10:15:00 | 1431.90 | 1437.68 | 1422.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1431.90 | 1437.68 | 1422.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 1431.90 | 1437.68 | 1422.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1426.35 | 1433.40 | 1422.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 1425.95 | 1433.40 | 1422.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 1427.00 | 1431.56 | 1424.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 1414.80 | 1431.56 | 1424.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 1399.40 | 1425.13 | 1422.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 1394.00 | 1425.13 | 1422.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1407.00 | 1421.50 | 1420.82 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 11:15:00 | 1410.50 | 1419.30 | 1419.88 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 1421.00 | 1418.00 | 1417.96 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 09:15:00 | 1399.20 | 1414.24 | 1416.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1387.60 | 1404.14 | 1410.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1401.00 | 1400.16 | 1406.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1401.00 | 1400.16 | 1406.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1401.00 | 1400.16 | 1406.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:30:00 | 1408.50 | 1400.16 | 1406.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1346.15 | 1298.41 | 1308.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:45:00 | 1349.90 | 1298.41 | 1308.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1348.40 | 1308.41 | 1312.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 1348.40 | 1308.41 | 1312.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1347.90 | 1316.31 | 1315.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 1353.70 | 1332.51 | 1323.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 1346.00 | 1346.79 | 1336.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:30:00 | 1347.95 | 1346.79 | 1336.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1342.05 | 1346.36 | 1341.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 1342.05 | 1346.36 | 1341.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1344.00 | 1345.89 | 1342.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1344.25 | 1345.89 | 1342.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1338.25 | 1344.36 | 1341.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:30:00 | 1338.85 | 1344.36 | 1341.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1340.00 | 1343.49 | 1341.65 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 15:15:00 | 1339.00 | 1340.43 | 1340.59 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 10:15:00 | 1346.35 | 1341.26 | 1340.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 1406.55 | 1357.50 | 1349.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 10:15:00 | 1390.35 | 1400.97 | 1382.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 11:00:00 | 1390.35 | 1400.97 | 1382.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1393.00 | 1394.69 | 1385.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:15:00 | 1362.60 | 1394.69 | 1385.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1367.80 | 1389.31 | 1384.15 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 12:15:00 | 1369.50 | 1379.32 | 1380.29 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1399.45 | 1382.94 | 1381.51 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1346.15 | 1382.04 | 1383.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 1323.45 | 1370.32 | 1378.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 1183.30 | 1168.21 | 1192.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 14:15:00 | 1183.30 | 1168.21 | 1192.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1183.30 | 1168.21 | 1192.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 1183.30 | 1168.21 | 1192.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1251.10 | 1186.84 | 1197.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:00:00 | 1251.10 | 1186.84 | 1197.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1268.85 | 1203.24 | 1203.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1268.85 | 1203.24 | 1203.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 1277.55 | 1218.10 | 1210.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 14:15:00 | 1295.65 | 1249.51 | 1228.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1299.50 | 1321.32 | 1290.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 1299.50 | 1321.32 | 1290.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1287.05 | 1314.46 | 1290.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 1287.45 | 1314.46 | 1290.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1298.80 | 1311.33 | 1291.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 15:00:00 | 1300.10 | 1304.25 | 1292.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:15:00 | 1300.00 | 1303.21 | 1295.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 1301.95 | 1302.24 | 1295.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 12:15:00 | 1310.00 | 1314.13 | 1314.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 1310.00 | 1314.13 | 1314.52 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-11-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 14:15:00 | 1358.95 | 1321.81 | 1317.86 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 14:15:00 | 1301.80 | 1333.04 | 1334.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1249.80 | 1312.05 | 1324.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1262.75 | 1257.35 | 1284.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 1262.75 | 1257.35 | 1284.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1256.00 | 1236.06 | 1249.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1256.00 | 1236.06 | 1249.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1259.50 | 1240.75 | 1250.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1258.95 | 1240.75 | 1250.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 14:15:00 | 1239.05 | 1246.13 | 1250.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:15:00 | 1233.00 | 1246.13 | 1250.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 10:15:00 | 1251.80 | 1231.00 | 1228.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1251.80 | 1231.00 | 1228.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 1265.90 | 1237.98 | 1231.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1286.15 | 1286.21 | 1273.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 1286.15 | 1286.21 | 1273.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1281.20 | 1286.81 | 1279.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:30:00 | 1281.95 | 1286.81 | 1279.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1327.05 | 1301.10 | 1289.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 13:45:00 | 1345.95 | 1318.83 | 1302.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 11:15:00 | 1355.45 | 1334.07 | 1316.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 1411.75 | 1415.10 | 1415.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 1411.75 | 1415.10 | 1415.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 15:15:00 | 1402.10 | 1412.50 | 1413.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 1377.75 | 1353.84 | 1368.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 1377.75 | 1353.84 | 1368.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1377.75 | 1353.84 | 1368.51 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1414.55 | 1377.17 | 1375.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1436.25 | 1395.04 | 1384.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 1496.00 | 1504.22 | 1483.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 1496.00 | 1504.22 | 1483.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1496.00 | 1504.22 | 1483.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 1489.05 | 1504.22 | 1483.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1489.45 | 1501.06 | 1488.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 1489.45 | 1501.06 | 1488.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1491.00 | 1499.05 | 1489.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:15:00 | 1492.90 | 1499.05 | 1489.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 1492.90 | 1497.82 | 1489.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 1480.85 | 1497.82 | 1489.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1491.40 | 1496.53 | 1489.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:45:00 | 1468.50 | 1496.53 | 1489.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 1502.90 | 1497.81 | 1490.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 1505.95 | 1497.81 | 1490.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 10:15:00 | 1479.05 | 1491.69 | 1491.16 | SL hit (close<static) qty=1.00 sl=1484.05 alert=retest2 |

### Cycle 50 — SELL (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 11:15:00 | 1478.00 | 1488.95 | 1489.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 12:15:00 | 1470.10 | 1485.18 | 1488.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 15:15:00 | 1458.60 | 1454.22 | 1465.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 09:15:00 | 1456.05 | 1454.22 | 1465.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1449.00 | 1453.17 | 1464.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1445.70 | 1451.54 | 1459.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1436.80 | 1451.23 | 1458.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 10:15:00 | 1513.85 | 1463.73 | 1462.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 1513.85 | 1463.73 | 1462.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 1519.35 | 1494.38 | 1479.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 15:15:00 | 1503.35 | 1504.58 | 1494.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 09:15:00 | 1567.50 | 1504.58 | 1494.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 1537.20 | 1545.55 | 1537.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 1537.20 | 1545.55 | 1537.20 | SL hit (close<ema400) qty=1.00 sl=1537.20 alert=retest1 |

### Cycle 52 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1520.20 | 1531.97 | 1532.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1477.85 | 1521.14 | 1527.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1483.35 | 1478.01 | 1497.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 1483.35 | 1478.01 | 1497.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1491.95 | 1483.28 | 1493.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:30:00 | 1497.00 | 1483.28 | 1493.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1490.90 | 1484.81 | 1493.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1457.90 | 1484.81 | 1493.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1459.00 | 1479.65 | 1490.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 10:30:00 | 1445.80 | 1472.92 | 1486.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 1446.00 | 1472.92 | 1486.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:30:00 | 1448.05 | 1466.80 | 1482.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1373.51 | 1405.51 | 1432.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1373.70 | 1405.51 | 1432.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1375.65 | 1405.51 | 1432.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 1303.24 | 1355.54 | 1390.99 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 1303.15 | 1284.72 | 1284.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 1320.85 | 1300.77 | 1294.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 1294.80 | 1307.89 | 1301.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 1294.80 | 1307.89 | 1301.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 1294.80 | 1307.89 | 1301.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 1294.80 | 1307.89 | 1301.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 1299.80 | 1306.27 | 1301.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 1302.55 | 1306.27 | 1301.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 1293.70 | 1303.76 | 1300.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:15:00 | 1292.40 | 1303.76 | 1300.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 1285.10 | 1297.96 | 1298.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1251.75 | 1286.73 | 1293.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1270.00 | 1265.18 | 1278.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 1270.00 | 1265.18 | 1278.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1298.10 | 1272.37 | 1279.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1298.10 | 1272.37 | 1279.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1299.45 | 1277.79 | 1281.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 1296.50 | 1277.79 | 1281.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 12:15:00 | 1302.00 | 1285.98 | 1284.45 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 1277.70 | 1284.40 | 1284.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1255.45 | 1273.82 | 1279.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 14:15:00 | 1243.95 | 1224.60 | 1244.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 14:15:00 | 1243.95 | 1224.60 | 1244.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 14:15:00 | 1243.95 | 1224.60 | 1244.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 15:00:00 | 1243.95 | 1224.60 | 1244.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 1264.90 | 1232.66 | 1246.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 1226.00 | 1232.66 | 1246.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1216.60 | 1229.45 | 1243.38 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 1259.10 | 1247.61 | 1246.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 11:15:00 | 1271.95 | 1252.48 | 1249.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1281.15 | 1290.10 | 1276.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 14:00:00 | 1281.15 | 1290.10 | 1276.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1282.20 | 1289.52 | 1279.38 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 1259.30 | 1280.53 | 1281.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 1254.90 | 1268.49 | 1274.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1259.35 | 1257.70 | 1265.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1259.35 | 1257.70 | 1265.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1259.35 | 1257.70 | 1265.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:45:00 | 1245.45 | 1254.67 | 1262.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 13:00:00 | 1246.50 | 1253.04 | 1261.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 10:00:00 | 1241.55 | 1243.32 | 1253.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 10:45:00 | 1242.85 | 1243.36 | 1252.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 14:15:00 | 1232.00 | 1237.53 | 1246.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 14:45:00 | 1235.00 | 1237.53 | 1246.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1237.10 | 1236.72 | 1244.53 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1252.00 | 1244.90 | 1244.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 12:15:00 | 1252.00 | 1244.90 | 1244.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 09:15:00 | 1269.15 | 1253.76 | 1249.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 15:15:00 | 1230.00 | 1259.97 | 1256.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 15:15:00 | 1230.00 | 1259.97 | 1256.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1230.00 | 1259.97 | 1256.35 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1221.70 | 1252.31 | 1253.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1212.50 | 1238.66 | 1246.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1208.65 | 1207.78 | 1223.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 13:00:00 | 1208.65 | 1207.78 | 1223.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1232.50 | 1207.88 | 1217.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 1232.50 | 1207.88 | 1217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1232.80 | 1212.86 | 1219.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 1239.95 | 1212.86 | 1219.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 14:15:00 | 1222.20 | 1218.81 | 1220.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 14:45:00 | 1226.00 | 1218.81 | 1220.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 15:15:00 | 1222.50 | 1219.55 | 1220.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 09:15:00 | 1205.80 | 1219.55 | 1220.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 1201.20 | 1199.22 | 1208.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:45:00 | 1215.90 | 1199.22 | 1208.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1154.05 | 1135.06 | 1154.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1154.05 | 1135.06 | 1154.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1158.95 | 1139.83 | 1154.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1172.25 | 1139.83 | 1154.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1172.75 | 1146.42 | 1156.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 12:00:00 | 1172.75 | 1146.42 | 1156.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1193.15 | 1162.88 | 1161.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 15:15:00 | 1202.00 | 1170.71 | 1165.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1198.40 | 1214.39 | 1197.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 1198.40 | 1214.39 | 1197.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1198.40 | 1214.39 | 1197.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1198.40 | 1214.39 | 1197.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1213.90 | 1214.29 | 1198.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 1221.60 | 1214.79 | 1203.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 15:15:00 | 1193.15 | 1201.78 | 1202.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 1193.15 | 1201.78 | 1202.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 1172.70 | 1195.96 | 1199.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1030.30 | 1026.57 | 1061.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 15:00:00 | 1030.30 | 1026.57 | 1061.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1062.30 | 1032.99 | 1058.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1062.30 | 1032.99 | 1058.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1054.65 | 1037.32 | 1057.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 1046.95 | 1039.25 | 1056.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:30:00 | 1048.40 | 1038.81 | 1055.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 1079.20 | 1056.70 | 1059.09 | SL hit (close>static) qty=1.00 sl=1070.70 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1079.70 | 1061.30 | 1060.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1084.65 | 1065.97 | 1063.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 13:15:00 | 1067.60 | 1068.60 | 1064.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 13:15:00 | 1067.60 | 1068.60 | 1064.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 13:15:00 | 1067.60 | 1068.60 | 1064.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-05 13:30:00 | 1067.95 | 1068.60 | 1064.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 14:15:00 | 1086.60 | 1072.20 | 1066.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 09:15:00 | 1106.00 | 1075.96 | 1069.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-12 09:15:00 | 1216.60 | 1160.40 | 1144.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 1211.85 | 1229.68 | 1232.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 1199.35 | 1210.97 | 1218.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 1210.15 | 1206.45 | 1214.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 1210.15 | 1206.45 | 1214.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1210.15 | 1206.45 | 1214.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 1211.85 | 1206.45 | 1214.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 1214.00 | 1207.96 | 1214.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 12:00:00 | 1214.00 | 1207.96 | 1214.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1251.75 | 1216.72 | 1217.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1251.75 | 1216.72 | 1217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 1258.80 | 1225.13 | 1221.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 13:15:00 | 1269.65 | 1261.19 | 1253.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1261.20 | 1284.27 | 1274.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1261.20 | 1284.27 | 1274.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1261.20 | 1284.27 | 1274.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 1261.20 | 1284.27 | 1274.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 1259.15 | 1279.24 | 1273.31 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 13:15:00 | 1260.90 | 1268.96 | 1269.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1168.95 | 1247.87 | 1259.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1207.90 | 1200.62 | 1223.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1207.90 | 1200.62 | 1223.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1207.90 | 1200.62 | 1223.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1190.15 | 1200.23 | 1221.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 1197.60 | 1202.88 | 1216.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1198.50 | 1202.91 | 1214.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 14:45:00 | 1200.70 | 1197.57 | 1206.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1220.70 | 1201.14 | 1206.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 1233.50 | 1213.96 | 1211.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1233.50 | 1213.96 | 1211.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1260.00 | 1229.01 | 1219.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 1248.10 | 1248.83 | 1239.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 12:30:00 | 1251.00 | 1248.83 | 1239.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1248.80 | 1248.82 | 1240.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 1240.90 | 1248.82 | 1240.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1237.40 | 1245.79 | 1240.95 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 13:15:00 | 1228.50 | 1237.12 | 1237.93 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 1261.00 | 1240.05 | 1238.89 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 13:15:00 | 1247.80 | 1252.25 | 1252.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 11:15:00 | 1245.20 | 1248.82 | 1250.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1215.00 | 1214.12 | 1226.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 1215.00 | 1214.12 | 1226.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1219.90 | 1211.73 | 1218.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 1204.00 | 1216.67 | 1218.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 10:15:00 | 1198.70 | 1214.53 | 1217.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1143.80 | 1167.54 | 1178.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 1166.80 | 1165.82 | 1175.31 | SL hit (close>ema200) qty=0.50 sl=1165.82 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1221.00 | 1181.83 | 1178.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 11:15:00 | 1231.00 | 1191.67 | 1183.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1191.00 | 1195.17 | 1186.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 1191.00 | 1195.17 | 1186.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1175.50 | 1191.24 | 1185.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 1171.90 | 1191.24 | 1185.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1178.00 | 1188.59 | 1185.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1153.50 | 1188.59 | 1185.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1159.90 | 1182.85 | 1182.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:45:00 | 1152.90 | 1182.85 | 1182.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-05-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 10:15:00 | 1159.20 | 1178.12 | 1180.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 11:15:00 | 1150.10 | 1172.52 | 1177.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1214.30 | 1171.32 | 1173.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1214.30 | 1171.32 | 1173.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1214.30 | 1171.32 | 1173.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 1214.30 | 1171.32 | 1173.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1227.00 | 1182.46 | 1178.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1257.80 | 1229.11 | 1221.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 10:15:00 | 1269.90 | 1271.39 | 1259.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 10:30:00 | 1269.10 | 1271.39 | 1259.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1258.30 | 1267.68 | 1259.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 1256.70 | 1267.68 | 1259.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1251.30 | 1264.41 | 1258.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1251.30 | 1264.41 | 1258.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1243.00 | 1260.13 | 1257.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:15:00 | 1254.00 | 1260.13 | 1257.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1251.80 | 1259.06 | 1257.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 1251.80 | 1259.06 | 1257.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1257.20 | 1258.69 | 1257.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 1260.70 | 1258.97 | 1257.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:45:00 | 1260.60 | 1260.36 | 1258.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 14:30:00 | 1267.30 | 1271.54 | 1266.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 14:15:00 | 1386.77 | 1329.78 | 1305.78 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1276.00 | 1301.86 | 1303.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1267.20 | 1294.93 | 1299.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1254.00 | 1253.53 | 1265.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:30:00 | 1259.60 | 1253.53 | 1265.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1249.50 | 1250.69 | 1259.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 1260.00 | 1250.69 | 1259.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 1266.90 | 1253.58 | 1259.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 1265.10 | 1253.58 | 1259.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1259.10 | 1254.69 | 1259.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 12:00:00 | 1256.10 | 1254.97 | 1258.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 1257.50 | 1255.48 | 1258.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1269.00 | 1260.47 | 1260.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1269.00 | 1260.47 | 1260.36 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 1252.80 | 1259.48 | 1260.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 1250.00 | 1255.19 | 1257.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1265.60 | 1256.78 | 1257.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1265.60 | 1256.78 | 1257.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1270.00 | 1259.42 | 1258.78 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1255.90 | 1259.05 | 1259.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1253.20 | 1257.88 | 1258.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1264.60 | 1251.30 | 1253.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1264.60 | 1251.30 | 1253.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1261.10 | 1253.26 | 1254.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:15:00 | 1258.60 | 1253.26 | 1254.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 1263.00 | 1256.41 | 1255.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 1263.00 | 1256.41 | 1255.56 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 14:15:00 | 1251.90 | 1256.62 | 1256.66 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 09:15:00 | 1260.80 | 1257.20 | 1256.90 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1253.20 | 1256.97 | 1256.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1242.90 | 1250.37 | 1253.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1201.20 | 1195.24 | 1208.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 1204.50 | 1195.24 | 1208.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1202.80 | 1196.75 | 1208.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 1206.20 | 1196.75 | 1208.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1208.60 | 1195.48 | 1202.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 1208.60 | 1195.48 | 1202.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1199.90 | 1196.36 | 1201.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 1196.70 | 1196.43 | 1201.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 1195.30 | 1196.66 | 1200.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 14:15:00 | 1195.20 | 1196.66 | 1200.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1195.00 | 1195.31 | 1199.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1177.40 | 1178.18 | 1183.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1177.40 | 1178.18 | 1183.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1180.00 | 1178.54 | 1183.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1168.70 | 1178.54 | 1183.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1207.50 | 1177.95 | 1179.12 | SL hit (close>static) qty=1.00 sl=1184.90 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 1208.30 | 1184.02 | 1181.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1224.10 | 1202.85 | 1193.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1221.20 | 1221.26 | 1215.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1221.20 | 1221.26 | 1215.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1212.00 | 1219.16 | 1215.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1212.00 | 1219.16 | 1215.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1214.10 | 1218.15 | 1215.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1229.00 | 1218.15 | 1215.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1218.80 | 1222.16 | 1218.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 1218.80 | 1222.16 | 1218.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1212.40 | 1220.21 | 1217.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 1212.40 | 1220.21 | 1217.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1221.20 | 1220.41 | 1218.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:30:00 | 1222.70 | 1218.50 | 1217.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 12:30:00 | 1222.20 | 1218.64 | 1217.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 14:15:00 | 1206.10 | 1215.92 | 1216.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 1206.10 | 1215.92 | 1216.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1200.90 | 1211.17 | 1214.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 1186.30 | 1185.84 | 1191.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 1186.30 | 1185.84 | 1191.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1190.90 | 1187.04 | 1191.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1192.00 | 1187.04 | 1191.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1190.40 | 1187.71 | 1191.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:30:00 | 1189.80 | 1187.71 | 1191.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1192.90 | 1188.46 | 1190.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1192.90 | 1188.46 | 1190.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1191.60 | 1189.09 | 1190.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1208.30 | 1189.09 | 1190.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 1208.00 | 1192.87 | 1192.12 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1194.20 | 1200.35 | 1200.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 1182.40 | 1195.03 | 1197.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 1141.00 | 1139.77 | 1146.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 1146.30 | 1139.77 | 1146.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1147.10 | 1141.24 | 1146.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 1147.90 | 1141.24 | 1146.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1143.90 | 1141.77 | 1146.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1140.70 | 1141.50 | 1145.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1141.90 | 1141.58 | 1144.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 1141.70 | 1143.83 | 1145.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1139.60 | 1142.98 | 1144.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1139.50 | 1140.10 | 1142.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 1139.50 | 1140.10 | 1142.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1144.90 | 1141.06 | 1142.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 1143.80 | 1141.06 | 1142.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1137.00 | 1140.25 | 1142.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:45:00 | 1134.20 | 1139.16 | 1141.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1083.66 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1084.81 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1084.62 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 09:15:00 | 1082.62 | 1099.87 | 1109.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 11:15:00 | 1100.00 | 1098.75 | 1107.36 | SL hit (close>ema200) qty=0.50 sl=1098.75 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 1123.00 | 1111.95 | 1111.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 1131.00 | 1115.76 | 1112.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1116.60 | 1127.05 | 1121.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1130.00 | 1126.86 | 1121.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 1129.20 | 1127.33 | 1122.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:30:00 | 1131.00 | 1128.34 | 1123.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1132.60 | 1129.20 | 1124.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1127.60 | 1129.44 | 1125.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 1125.70 | 1129.44 | 1125.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1121.80 | 1127.91 | 1125.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1121.80 | 1127.91 | 1125.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1120.50 | 1126.43 | 1124.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:30:00 | 1120.80 | 1126.43 | 1124.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 1118.30 | 1123.60 | 1123.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1114.90 | 1121.86 | 1122.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1101.90 | 1083.77 | 1088.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 1114.80 | 1083.77 | 1088.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1108.00 | 1088.62 | 1090.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1105.50 | 1088.62 | 1090.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1091.90 | 1090.14 | 1090.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 1083.20 | 1089.93 | 1090.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 1029.04 | 1071.87 | 1081.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-12 09:15:00 | 974.88 | 1013.39 | 1042.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 970.30 | 955.79 | 955.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 985.60 | 964.92 | 959.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 989.50 | 995.67 | 986.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 09:30:00 | 992.70 | 995.67 | 986.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 990.10 | 996.36 | 991.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 990.10 | 996.36 | 991.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 984.00 | 993.88 | 990.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 984.00 | 993.88 | 990.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 983.00 | 988.07 | 988.67 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 1040.50 | 997.83 | 992.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 10:15:00 | 1068.80 | 1012.02 | 999.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 10:15:00 | 1056.10 | 1056.68 | 1034.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:00:00 | 1056.10 | 1056.68 | 1034.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1045.00 | 1051.85 | 1040.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1036.10 | 1051.85 | 1040.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1079.70 | 1057.42 | 1044.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 1095.90 | 1057.42 | 1044.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 11:30:00 | 1086.00 | 1065.61 | 1050.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 1091.20 | 1070.86 | 1064.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 14:45:00 | 1082.60 | 1078.61 | 1071.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1087.50 | 1080.88 | 1073.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 1097.40 | 1086.25 | 1078.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:45:00 | 1099.70 | 1088.40 | 1079.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1097.40 | 1088.40 | 1079.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1097.20 | 1090.79 | 1083.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1083.30 | 1089.29 | 1083.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1083.30 | 1089.29 | 1083.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1073.80 | 1086.19 | 1082.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1073.80 | 1086.19 | 1082.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1074.50 | 1083.86 | 1081.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:45:00 | 1072.90 | 1083.86 | 1081.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1072.80 | 1080.56 | 1080.52 | SL hit (close<static) qty=1.00 sl=1073.50 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1071.00 | 1078.65 | 1079.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1059.70 | 1074.86 | 1077.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 15:15:00 | 1063.00 | 1062.10 | 1068.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 09:15:00 | 1065.70 | 1062.10 | 1068.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1075.20 | 1064.72 | 1069.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1075.20 | 1064.72 | 1069.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1070.00 | 1065.78 | 1069.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 1067.10 | 1066.94 | 1069.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 09:15:00 | 1108.40 | 1073.10 | 1071.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1108.40 | 1073.10 | 1071.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 1139.50 | 1086.38 | 1077.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 13:15:00 | 1125.80 | 1127.24 | 1111.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 14:00:00 | 1125.80 | 1127.24 | 1111.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1114.90 | 1123.03 | 1115.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 1114.90 | 1123.03 | 1115.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1111.30 | 1120.68 | 1115.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1111.30 | 1120.68 | 1115.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1110.40 | 1118.62 | 1115.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1111.20 | 1118.62 | 1115.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1121.00 | 1118.33 | 1115.46 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1111.00 | 1119.30 | 1119.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 1110.60 | 1117.56 | 1118.70 | Break + close below crossover candle low |

### Cycle 95 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 1136.90 | 1119.96 | 1119.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1157.50 | 1136.84 | 1130.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 1144.80 | 1152.98 | 1146.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 15:00:00 | 1144.80 | 1152.98 | 1146.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 1150.00 | 1152.39 | 1146.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1161.00 | 1152.39 | 1146.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 12:30:00 | 1151.00 | 1150.66 | 1147.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:30:00 | 1151.00 | 1150.89 | 1148.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 10:15:00 | 1140.80 | 1149.49 | 1148.44 | SL hit (close<static) qty=1.00 sl=1142.40 alert=retest2 |

### Cycle 96 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1139.40 | 1147.47 | 1147.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 1135.00 | 1141.82 | 1144.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 1074.10 | 1071.96 | 1081.56 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-01 09:15:00 | 1062.50 | 1071.96 | 1081.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1073.80 | 1066.87 | 1072.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 1073.80 | 1066.87 | 1072.92 | SL hit (close>ema400) qty=1.00 sl=1072.92 alert=retest1 |

### Cycle 97 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1088.90 | 1076.36 | 1074.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 15:15:00 | 1091.60 | 1084.64 | 1080.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 1084.10 | 1084.54 | 1080.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 1084.10 | 1084.54 | 1080.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1081.80 | 1083.99 | 1080.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 1081.80 | 1083.99 | 1080.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1081.60 | 1083.51 | 1080.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:30:00 | 1082.30 | 1083.51 | 1080.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 1081.30 | 1083.07 | 1080.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 1082.60 | 1083.07 | 1080.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1078.40 | 1082.09 | 1080.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1078.40 | 1082.09 | 1080.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1081.90 | 1082.05 | 1080.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1077.30 | 1082.05 | 1080.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1075.90 | 1080.82 | 1080.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 1075.00 | 1080.82 | 1080.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1070.70 | 1078.80 | 1079.54 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 1083.00 | 1079.64 | 1079.43 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1075.70 | 1078.85 | 1079.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 10:15:00 | 1074.50 | 1077.98 | 1078.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1084.50 | 1073.33 | 1074.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 1088.10 | 1073.33 | 1074.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1079.90 | 1074.65 | 1074.68 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 1079.60 | 1075.64 | 1075.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 1084.70 | 1078.21 | 1076.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 15:15:00 | 1077.30 | 1078.67 | 1076.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 09:15:00 | 1092.80 | 1078.67 | 1076.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1098.00 | 1082.53 | 1078.90 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 1074.30 | 1080.38 | 1080.99 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1087.40 | 1081.91 | 1081.50 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 12:15:00 | 1076.00 | 1080.97 | 1081.15 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 10:15:00 | 1085.80 | 1080.84 | 1080.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 1094.40 | 1085.15 | 1082.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 11:15:00 | 1108.70 | 1109.49 | 1101.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 12:00:00 | 1108.70 | 1109.49 | 1101.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1102.60 | 1108.40 | 1102.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1102.60 | 1108.40 | 1102.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1101.00 | 1106.92 | 1102.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1119.00 | 1106.92 | 1102.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 12:15:00 | 1087.90 | 1106.55 | 1108.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 1087.90 | 1106.55 | 1108.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 15:15:00 | 1080.00 | 1095.23 | 1102.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 13:15:00 | 1095.00 | 1093.46 | 1098.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 14:00:00 | 1095.00 | 1093.46 | 1098.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1104.50 | 1095.67 | 1099.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 1104.50 | 1095.67 | 1099.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 1110.00 | 1098.54 | 1100.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 1109.90 | 1098.54 | 1100.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1116.30 | 1102.09 | 1101.61 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1101.50 | 1106.20 | 1106.36 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1108.70 | 1106.70 | 1106.57 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1096.80 | 1104.72 | 1105.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1093.70 | 1102.52 | 1104.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 13:15:00 | 1095.00 | 1093.16 | 1097.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 1095.00 | 1093.16 | 1097.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1097.70 | 1094.07 | 1097.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1097.70 | 1094.07 | 1097.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1099.00 | 1095.05 | 1097.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1086.20 | 1095.05 | 1097.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1081.20 | 1092.28 | 1095.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:30:00 | 1079.50 | 1088.37 | 1093.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1025.52 | 1066.40 | 1079.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1004.10 | 1001.30 | 1013.26 | SL hit (close>ema200) qty=0.50 sl=1001.30 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 980.50 | 968.32 | 968.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 15:15:00 | 985.00 | 979.75 | 976.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 986.80 | 988.73 | 983.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 986.80 | 988.73 | 983.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 980.30 | 986.40 | 983.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 980.30 | 986.40 | 983.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 981.00 | 985.32 | 983.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 982.45 | 984.56 | 982.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 970.60 | 980.77 | 981.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 970.60 | 980.77 | 981.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 09:15:00 | 968.55 | 973.51 | 975.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 974.80 | 972.29 | 974.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 974.80 | 972.29 | 974.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 974.95 | 972.83 | 974.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 975.00 | 972.83 | 974.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 980.80 | 974.42 | 975.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 966.45 | 974.42 | 975.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 918.13 | 932.07 | 947.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 12:15:00 | 931.85 | 930.49 | 942.86 | SL hit (close>ema200) qty=0.50 sl=930.49 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 974.65 | 938.92 | 934.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 981.80 | 947.49 | 938.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 975.30 | 976.57 | 961.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:45:00 | 973.30 | 976.57 | 961.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 961.50 | 973.83 | 967.07 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 950.00 | 963.10 | 963.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 946.60 | 957.83 | 960.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 942.00 | 929.43 | 935.26 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 951.25 | 938.92 | 938.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 953.95 | 941.93 | 939.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 11:15:00 | 962.00 | 964.63 | 957.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:00:00 | 962.00 | 964.63 | 957.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 954.40 | 960.60 | 957.75 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 949.65 | 955.78 | 956.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 946.05 | 952.76 | 954.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 943.45 | 933.54 | 939.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 943.45 | 933.54 | 939.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 925.95 | 932.02 | 938.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:30:00 | 945.90 | 932.02 | 938.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 937.50 | 932.60 | 937.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 937.50 | 932.60 | 937.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 939.80 | 934.04 | 937.86 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 952.30 | 941.04 | 940.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 965.25 | 948.47 | 945.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 959.35 | 964.06 | 956.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 959.35 | 964.06 | 956.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 955.00 | 962.24 | 956.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 955.00 | 962.24 | 956.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 955.25 | 960.85 | 956.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:45:00 | 955.00 | 960.85 | 956.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 954.00 | 959.48 | 956.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 954.35 | 959.48 | 956.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 952.00 | 957.98 | 955.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 952.00 | 957.98 | 955.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 953.05 | 956.09 | 955.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 941.00 | 956.09 | 955.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 942.40 | 953.36 | 954.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 10:15:00 | 935.60 | 949.80 | 952.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 935.20 | 933.49 | 939.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 928.45 | 933.49 | 939.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 941.85 | 935.16 | 939.26 | SL hit (close>static) qty=1.00 sl=940.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 819.65 | 806.95 | 805.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 829.20 | 813.88 | 808.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 814.85 | 817.85 | 811.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 814.85 | 817.85 | 811.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 809.45 | 816.17 | 811.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 810.00 | 816.17 | 811.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 827.85 | 818.51 | 813.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:45:00 | 811.00 | 818.51 | 813.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 844.00 | 832.99 | 823.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 848.70 | 836.57 | 825.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 846.05 | 839.59 | 828.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 848.35 | 842.19 | 831.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 09:15:00 | 933.57 | 912.51 | 899.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 908.95 | 914.17 | 914.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 907.20 | 912.77 | 913.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 899.00 | 897.26 | 903.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 13:45:00 | 900.15 | 897.26 | 903.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 904.60 | 898.80 | 902.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:30:00 | 904.20 | 898.80 | 902.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 905.45 | 900.13 | 902.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:30:00 | 904.45 | 900.13 | 902.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 894.05 | 898.91 | 902.16 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 908.00 | 904.13 | 903.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 921.40 | 907.58 | 905.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 904.30 | 914.04 | 910.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 904.30 | 914.04 | 910.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 899.05 | 911.04 | 909.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 899.05 | 911.04 | 909.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 11:15:00 | 898.35 | 908.51 | 908.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 15:15:00 | 897.00 | 902.52 | 905.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 877.80 | 871.94 | 880.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 881.45 | 871.94 | 880.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 874.40 | 872.79 | 878.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 875.45 | 872.79 | 878.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 880.00 | 874.94 | 878.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 880.00 | 874.94 | 878.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 876.00 | 875.15 | 878.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 13:15:00 | 865.70 | 872.81 | 876.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 15:00:00 | 865.45 | 870.23 | 874.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 866.50 | 869.75 | 873.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 909.00 | 878.27 | 875.66 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 875.15 | 882.06 | 882.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 868.75 | 879.40 | 881.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 866.95 | 859.80 | 866.19 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 887.00 | 870.26 | 868.72 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 847.65 | 867.45 | 869.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 839.15 | 852.02 | 855.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 853.30 | 852.00 | 855.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:45:00 | 853.65 | 852.00 | 855.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 884.00 | 858.40 | 857.75 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 842.95 | 855.78 | 857.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 841.60 | 851.33 | 854.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 828.00 | 823.41 | 833.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 828.00 | 823.41 | 833.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 819.45 | 825.46 | 831.06 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 843.35 | 834.32 | 833.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 844.40 | 836.34 | 834.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 837.65 | 839.38 | 836.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 849.40 | 839.82 | 837.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:30:00 | 847.80 | 841.23 | 838.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 833.30 | 837.39 | 837.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 833.30 | 837.39 | 837.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 829.30 | 835.77 | 836.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 802.35 | 797.65 | 809.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 802.75 | 797.65 | 809.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 834.00 | 805.42 | 809.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 834.00 | 805.42 | 809.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 830.00 | 810.33 | 811.06 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 828.25 | 813.92 | 812.62 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 793.35 | 812.92 | 813.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 782.90 | 806.92 | 810.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 795.00 | 768.83 | 780.05 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 812.35 | 786.77 | 786.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 819.10 | 802.02 | 795.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 11:15:00 | 893.00 | 894.80 | 880.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 893.00 | 894.80 | 880.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 919.80 | 916.60 | 911.33 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 902.50 | 910.42 | 910.44 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 928.80 | 914.10 | 912.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 936.15 | 919.16 | 915.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 931.20 | 932.49 | 926.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 10:15:00 | 928.50 | 931.70 | 926.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 928.50 | 931.70 | 926.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 925.15 | 931.70 | 926.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 927.00 | 930.76 | 926.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 925.50 | 930.76 | 926.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 927.00 | 930.01 | 926.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 923.55 | 930.01 | 926.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 922.60 | 928.52 | 926.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 922.60 | 928.52 | 926.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 915.95 | 926.01 | 925.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 915.95 | 926.01 | 925.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 918.30 | 924.47 | 924.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 912.25 | 920.67 | 922.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 913.70 | 912.35 | 916.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 916.50 | 912.35 | 916.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 915.40 | 912.96 | 916.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 916.20 | 912.96 | 916.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 915.80 | 913.53 | 916.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 915.80 | 913.53 | 916.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 916.85 | 914.19 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 916.85 | 914.19 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 916.80 | 914.71 | 916.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 916.80 | 914.71 | 916.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 916.70 | 915.11 | 916.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 15:00:00 | 916.70 | 915.11 | 916.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 916.55 | 915.40 | 916.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 919.20 | 915.40 | 916.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 910.95 | 914.51 | 916.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 907.10 | 913.81 | 915.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 09:45:00 | 905.90 | 907.94 | 911.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 905.80 | 890.43 | 889.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 905.80 | 890.43 | 889.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 934.75 | 906.81 | 900.18 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-17 09:15:00 | 1434.30 | 2024-05-21 14:15:00 | 1577.73 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-05 14:00:00 | 1350.00 | 2024-06-06 12:15:00 | 1403.95 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2024-06-11 09:15:00 | 1486.45 | 2024-06-11 09:15:00 | 1430.00 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2024-06-13 13:15:00 | 1455.25 | 2024-06-19 12:15:00 | 1453.85 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-06-13 15:15:00 | 1456.00 | 2024-06-19 12:15:00 | 1453.85 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-06-19 09:30:00 | 1462.20 | 2024-06-19 12:15:00 | 1453.85 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-06-19 11:00:00 | 1457.10 | 2024-06-19 12:15:00 | 1453.85 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-07-02 12:45:00 | 1473.45 | 2024-07-02 15:15:00 | 1493.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-07-10 10:15:00 | 1441.25 | 2024-07-15 15:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-07-10 11:15:00 | 1440.00 | 2024-07-15 15:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-07-11 09:45:00 | 1445.00 | 2024-07-15 15:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-07-11 12:00:00 | 1435.00 | 2024-07-15 15:15:00 | 1447.30 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1355.00 | 2024-07-24 09:15:00 | 1413.10 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2024-07-23 13:00:00 | 1379.30 | 2024-07-24 09:15:00 | 1413.10 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-07-23 13:30:00 | 1377.75 | 2024-07-24 09:15:00 | 1413.10 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-07-29 14:00:00 | 1389.25 | 2024-07-30 10:15:00 | 1441.10 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2024-08-06 12:00:00 | 1293.05 | 2024-08-14 14:15:00 | 1232.72 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2024-08-07 13:15:00 | 1297.60 | 2024-08-14 14:15:00 | 1234.00 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2024-08-08 14:15:00 | 1298.95 | 2024-08-14 14:15:00 | 1234.33 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2024-08-08 14:45:00 | 1299.30 | 2024-08-14 14:15:00 | 1236.00 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1301.05 | 2024-08-14 14:15:00 | 1232.86 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1297.75 | 2024-08-14 14:15:00 | 1234.76 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2024-08-12 09:45:00 | 1294.50 | 2024-08-14 14:15:00 | 1234.90 | PARTIAL | 0.50 | 4.60% |
| SELL | retest2 | 2024-08-12 12:00:00 | 1299.75 | 2024-08-14 15:15:00 | 1228.40 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2024-08-12 14:15:00 | 1299.90 | 2024-08-14 15:15:00 | 1229.77 | PARTIAL | 0.50 | 5.39% |
| SELL | retest2 | 2024-08-06 12:00:00 | 1293.05 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.41% |
| SELL | retest2 | 2024-08-07 13:15:00 | 1297.60 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.75% |
| SELL | retest2 | 2024-08-08 14:15:00 | 1298.95 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2024-08-08 14:45:00 | 1299.30 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1301.05 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1297.75 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2024-08-12 09:45:00 | 1294.50 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2024-08-12 12:00:00 | 1299.75 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.91% |
| SELL | retest2 | 2024-08-12 14:15:00 | 1299.90 | 2024-08-16 09:15:00 | 1261.95 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2024-08-13 09:15:00 | 1287.95 | 2024-08-19 09:15:00 | 1299.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-08-21 14:30:00 | 1318.60 | 2024-08-23 14:15:00 | 1306.70 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-08-22 13:45:00 | 1318.30 | 2024-08-23 14:15:00 | 1306.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-08-22 15:00:00 | 1321.30 | 2024-08-23 14:15:00 | 1306.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-08-27 13:15:00 | 1298.75 | 2024-08-30 15:15:00 | 1294.60 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-08-27 14:00:00 | 1296.00 | 2024-08-30 15:15:00 | 1294.60 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2024-08-28 09:15:00 | 1288.90 | 2024-08-30 15:15:00 | 1294.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-09-10 11:45:00 | 1246.15 | 2024-09-11 10:15:00 | 1260.65 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-09-10 13:30:00 | 1245.95 | 2024-09-11 10:15:00 | 1260.65 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-04 15:00:00 | 1300.10 | 2024-11-08 12:15:00 | 1310.00 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2024-11-05 12:15:00 | 1300.00 | 2024-11-08 12:15:00 | 1310.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2024-11-05 13:15:00 | 1301.95 | 2024-11-08 12:15:00 | 1310.00 | STOP_HIT | 1.00 | 0.62% |
| SELL | retest2 | 2024-11-19 15:15:00 | 1233.00 | 2024-11-25 10:15:00 | 1251.80 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-12-02 13:45:00 | 1345.95 | 2024-12-11 14:15:00 | 1411.75 | STOP_HIT | 1.00 | 4.89% |
| BUY | retest2 | 2024-12-03 11:15:00 | 1355.45 | 2024-12-11 14:15:00 | 1411.75 | STOP_HIT | 1.00 | 4.15% |
| BUY | retest2 | 2024-12-23 11:15:00 | 1505.95 | 2024-12-24 10:15:00 | 1479.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-12-27 14:45:00 | 1445.70 | 2024-12-30 10:15:00 | 1513.85 | STOP_HIT | 1.00 | -4.71% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1436.80 | 2024-12-30 10:15:00 | 1513.85 | STOP_HIT | 1.00 | -5.36% |
| BUY | retest1 | 2025-01-01 09:15:00 | 1567.50 | 2025-01-03 12:15:00 | 1537.20 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1445.80 | 2025-01-10 09:15:00 | 1373.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1446.00 | 2025-01-10 09:15:00 | 1373.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:30:00 | 1448.05 | 2025-01-10 09:15:00 | 1375.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 10:30:00 | 1445.80 | 2025-01-13 09:15:00 | 1303.24 | TARGET_HIT | 0.50 | 9.86% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1446.00 | 2025-01-13 10:15:00 | 1301.22 | TARGET_HIT | 0.50 | 10.01% |
| SELL | retest2 | 2025-01-08 11:30:00 | 1448.05 | 2025-01-13 10:15:00 | 1301.40 | TARGET_HIT | 0.50 | 10.13% |
| SELL | retest2 | 2025-02-04 11:45:00 | 1245.45 | 2025-02-07 12:15:00 | 1252.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-02-04 13:00:00 | 1246.50 | 2025-02-07 12:15:00 | 1252.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-02-05 10:00:00 | 1241.55 | 2025-02-07 12:15:00 | 1252.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-02-05 10:45:00 | 1242.85 | 2025-02-07 12:15:00 | 1252.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-02-21 14:45:00 | 1221.60 | 2025-02-24 15:15:00 | 1193.15 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-03-04 12:00:00 | 1046.95 | 2025-03-05 09:15:00 | 1079.20 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-03-04 12:30:00 | 1048.40 | 2025-03-05 09:15:00 | 1079.20 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-03-06 09:15:00 | 1106.00 | 2025-03-12 09:15:00 | 1216.60 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1190.15 | 2025-04-11 12:15:00 | 1233.50 | STOP_HIT | 1.00 | -3.64% |
| SELL | retest2 | 2025-04-08 15:00:00 | 1197.60 | 2025-04-11 12:15:00 | 1233.50 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1198.50 | 2025-04-11 12:15:00 | 1233.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-04-09 14:45:00 | 1200.70 | 2025-04-11 12:15:00 | 1233.50 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-04-30 09:15:00 | 1204.00 | 2025-05-07 09:15:00 | 1143.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-30 09:15:00 | 1204.00 | 2025-05-07 11:15:00 | 1166.80 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2025-04-30 10:15:00 | 1198.70 | 2025-05-08 10:15:00 | 1221.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-05-21 13:30:00 | 1260.70 | 2025-05-26 14:15:00 | 1386.77 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-21 14:45:00 | 1260.60 | 2025-05-26 14:15:00 | 1386.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-22 14:30:00 | 1267.30 | 2025-05-28 09:15:00 | 1276.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-06-02 12:00:00 | 1256.10 | 2025-06-03 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-06-02 13:00:00 | 1257.50 | 2025-06-03 09:15:00 | 1269.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-09 11:15:00 | 1258.60 | 2025-06-09 14:15:00 | 1263.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-06-18 12:00:00 | 1196.70 | 2025-06-24 09:15:00 | 1207.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-18 13:45:00 | 1195.30 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-06-18 14:15:00 | 1195.20 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-06-19 09:30:00 | 1195.00 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1168.70 | 2025-06-24 10:15:00 | 1208.30 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2025-07-01 09:30:00 | 1222.70 | 2025-07-01 14:15:00 | 1206.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-01 12:30:00 | 1222.20 | 2025-07-01 14:15:00 | 1206.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1140.70 | 2025-07-29 09:15:00 | 1083.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1141.90 | 2025-07-29 09:15:00 | 1084.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1141.70 | 2025-07-29 09:15:00 | 1084.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1139.60 | 2025-07-29 09:15:00 | 1082.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1140.70 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-07-22 14:00:00 | 1141.90 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-07-23 09:15:00 | 1141.70 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1139.60 | 2025-07-29 11:15:00 | 1100.00 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-07-24 10:45:00 | 1134.20 | 2025-07-30 09:15:00 | 1123.00 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1130.00 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-07-31 12:00:00 | 1129.20 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-07-31 12:30:00 | 1131.00 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1132.60 | 2025-08-01 13:15:00 | 1118.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-08 12:15:00 | 1083.20 | 2025-08-11 09:15:00 | 1029.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 12:15:00 | 1083.20 | 2025-08-12 09:15:00 | 974.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-28 10:15:00 | 1095.90 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-08-28 11:30:00 | 1086.00 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1091.20 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-02 14:45:00 | 1082.60 | 2025-09-04 14:15:00 | 1072.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-09-03 13:00:00 | 1097.40 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-03 13:45:00 | 1099.70 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-09-03 14:15:00 | 1097.40 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1097.20 | 2025-09-04 15:15:00 | 1071.00 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-08 13:00:00 | 1067.10 | 2025-09-09 09:15:00 | 1108.40 | STOP_HIT | 1.00 | -3.87% |
| BUY | retest2 | 2025-09-23 09:15:00 | 1161.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-09-23 12:30:00 | 1151.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-09-23 13:30:00 | 1151.00 | 2025-09-24 10:15:00 | 1140.80 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest1 | 2025-10-01 09:15:00 | 1062.50 | 2025-10-01 15:15:00 | 1073.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1119.00 | 2025-10-28 12:15:00 | 1087.90 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1079.50 | 2025-11-07 09:15:00 | 1025.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:30:00 | 1079.50 | 2025-11-12 09:15:00 | 1004.10 | STOP_HIT | 0.50 | 6.98% |
| BUY | retest2 | 2025-12-01 13:30:00 | 982.45 | 2025-12-02 09:15:00 | 970.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-12-05 09:15:00 | 966.45 | 2025-12-09 09:15:00 | 918.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:15:00 | 966.45 | 2025-12-09 12:15:00 | 931.85 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-01-08 09:15:00 | 928.45 | 2026-01-08 09:15:00 | 941.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-01-08 11:45:00 | 930.05 | 2026-01-12 09:15:00 | 883.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 12:30:00 | 930.20 | 2026-01-12 09:15:00 | 883.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 14:15:00 | 929.05 | 2026-01-12 09:15:00 | 882.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 930.05 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-08 12:30:00 | 930.20 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2026-01-08 14:15:00 | 929.05 | 2026-01-14 13:15:00 | 884.90 | STOP_HIT | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-09 09:15:00 | 912.65 | 2026-01-16 09:15:00 | 867.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 09:15:00 | 912.65 | 2026-01-20 09:15:00 | 821.38 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-01-30 10:30:00 | 848.70 | 2026-02-09 09:15:00 | 933.57 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 12:45:00 | 846.05 | 2026-02-09 09:15:00 | 930.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 13:30:00 | 848.35 | 2026-02-09 09:15:00 | 933.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-24 13:15:00 | 865.70 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2026-02-24 15:00:00 | 865.45 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2026-02-25 09:15:00 | 866.50 | 2026-02-26 09:15:00 | 909.00 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2026-03-19 10:30:00 | 849.40 | 2026-03-20 13:15:00 | 833.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-03-19 12:30:00 | 847.80 | 2026-03-20 13:15:00 | 833.30 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-28 11:15:00 | 907.10 | 2026-05-07 09:15:00 | 905.80 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2026-04-29 09:45:00 | 905.90 | 2026-05-07 09:15:00 | 905.80 | STOP_HIT | 1.00 | 0.01% |
