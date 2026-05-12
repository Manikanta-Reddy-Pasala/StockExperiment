# DOMS Industries Ltd. (DOMS)

## Backtest Summary

- **Window:** 2023-12-20 09:15:00 → 2026-05-11 15:15:00 (4119 bars)
- **Last close:** 2320.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 174 |
| ALERT1 | 119 |
| ALERT2 | 118 |
| ALERT2_SKIP | 66 |
| ALERT3 | 297 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 146 |
| PARTIAL | 26 |
| TARGET_HIT | 14 |
| STOP_HIT | 138 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 178 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 81 / 97
- **Target hits / Stop hits / Partials:** 14 / 138 / 26
- **Avg / median % per leg:** 0.91% / -0.59%
- **Sum % (uncompounded):** 161.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 21 | 31.8% | 6 | 59 | 1 | -0.04% | -2.8% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.07% | 8.1% |
| BUY @ 3rd Alert (retest2) | 64 | 19 | 29.7% | 6 | 58 | 0 | -0.17% | -10.9% |
| SELL (all) | 112 | 60 | 53.6% | 8 | 79 | 25 | 1.46% | 163.9% |
| SELL @ 2nd Alert (retest1) | 8 | 6 | 75.0% | 0 | 5 | 3 | 2.85% | 22.8% |
| SELL @ 3rd Alert (retest2) | 104 | 54 | 51.9% | 8 | 74 | 22 | 1.36% | 141.1% |
| retest1 (combined) | 10 | 8 | 80.0% | 0 | 6 | 4 | 3.09% | 30.9% |
| retest2 (combined) | 168 | 73 | 43.5% | 14 | 132 | 22 | 0.78% | 130.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 12:15:00 | 1280.65 | 1274.81 | 1274.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 10:15:00 | 1286.80 | 1279.92 | 1277.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 15:15:00 | 1285.05 | 1286.19 | 1281.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 09:15:00 | 1300.90 | 1286.19 | 1281.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1324.95 | 1316.23 | 1309.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 11:00:00 | 1328.75 | 1318.22 | 1312.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 09:15:00 | 1365.95 | 1337.58 | 1326.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-01-10 12:15:00 | 1341.75 | 1343.06 | 1331.94 | SL hit (close<ema200) qty=0.50 sl=1343.06 alert=retest1 |

### Cycle 2 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 1436.05 | 1472.04 | 1476.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 1413.00 | 1443.32 | 1457.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 11:15:00 | 1446.25 | 1443.42 | 1455.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 11:15:00 | 1446.25 | 1443.42 | 1455.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 1446.25 | 1443.42 | 1455.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 1446.25 | 1443.42 | 1455.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1439.05 | 1437.12 | 1447.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 09:45:00 | 1444.05 | 1437.12 | 1447.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 10:15:00 | 1448.05 | 1439.31 | 1447.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 10:30:00 | 1449.05 | 1439.31 | 1447.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 11:15:00 | 1450.95 | 1441.63 | 1447.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-19 11:45:00 | 1452.00 | 1441.63 | 1447.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 1419.95 | 1433.96 | 1441.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 14:30:00 | 1410.85 | 1425.50 | 1430.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 09:15:00 | 1414.45 | 1425.84 | 1430.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 10:00:00 | 1412.35 | 1423.14 | 1428.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 13:00:00 | 1417.40 | 1418.04 | 1424.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 14:15:00 | 1420.60 | 1418.50 | 1423.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:45:00 | 1422.10 | 1418.50 | 1423.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1417.80 | 1418.92 | 1423.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 1412.05 | 1418.92 | 1423.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 11:30:00 | 1412.95 | 1416.13 | 1420.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 12:45:00 | 1411.75 | 1415.24 | 1420.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 13:45:00 | 1414.35 | 1417.05 | 1420.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 14:15:00 | 1452.95 | 1424.23 | 1423.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 1452.95 | 1424.23 | 1423.45 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 13:15:00 | 1418.10 | 1434.58 | 1434.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 14:15:00 | 1395.00 | 1426.67 | 1431.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 13:15:00 | 1413.25 | 1411.22 | 1419.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-31 14:00:00 | 1413.25 | 1411.22 | 1419.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 14:15:00 | 1403.80 | 1409.74 | 1418.37 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-01 15:15:00 | 1425.00 | 1419.79 | 1419.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 09:15:00 | 1439.65 | 1423.76 | 1421.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-02 12:15:00 | 1418.95 | 1424.78 | 1422.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 12:15:00 | 1418.95 | 1424.78 | 1422.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 12:15:00 | 1418.95 | 1424.78 | 1422.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 12:30:00 | 1420.95 | 1424.78 | 1422.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 13:15:00 | 1414.95 | 1422.81 | 1422.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:00:00 | 1414.95 | 1422.81 | 1422.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 14:15:00 | 1415.45 | 1421.34 | 1421.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 09:15:00 | 1410.00 | 1418.54 | 1420.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 14:15:00 | 1417.90 | 1414.96 | 1417.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 14:15:00 | 1417.90 | 1414.96 | 1417.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 14:15:00 | 1417.90 | 1414.96 | 1417.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 15:00:00 | 1417.90 | 1414.96 | 1417.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 15:15:00 | 1410.20 | 1414.01 | 1416.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 09:15:00 | 1425.00 | 1414.01 | 1416.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 1423.75 | 1415.96 | 1417.34 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 11:15:00 | 1434.10 | 1421.28 | 1419.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 14:15:00 | 1463.15 | 1444.17 | 1434.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 15:15:00 | 1550.00 | 1550.56 | 1517.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 1491.65 | 1538.78 | 1515.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1491.65 | 1538.78 | 1515.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-12 10:00:00 | 1491.65 | 1538.78 | 1515.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 10:15:00 | 1492.15 | 1529.45 | 1513.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 13:30:00 | 1509.90 | 1512.00 | 1507.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 15:15:00 | 1526.00 | 1565.45 | 1569.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1526.00 | 1565.45 | 1569.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1482.85 | 1548.93 | 1561.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 12:15:00 | 1536.20 | 1535.69 | 1551.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:00:00 | 1536.20 | 1535.69 | 1551.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 1533.80 | 1529.51 | 1540.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 11:30:00 | 1530.40 | 1529.51 | 1540.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 1541.95 | 1532.00 | 1540.69 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 09:15:00 | 1588.55 | 1552.98 | 1548.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-26 13:15:00 | 1604.05 | 1578.29 | 1563.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-27 11:15:00 | 1605.00 | 1607.36 | 1585.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 11:45:00 | 1608.20 | 1607.36 | 1585.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 12:15:00 | 1592.60 | 1604.41 | 1586.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:00:00 | 1592.60 | 1604.41 | 1586.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 13:15:00 | 1588.85 | 1601.30 | 1586.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 13:45:00 | 1593.00 | 1601.30 | 1586.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1593.05 | 1599.65 | 1586.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-28 09:15:00 | 1611.85 | 1597.72 | 1587.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 1557.60 | 1586.86 | 1584.74 | SL hit (close<static) qty=1.00 sl=1577.05 alert=retest2 |

### Cycle 10 — SELL (started 2024-02-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 12:15:00 | 1547.05 | 1578.90 | 1581.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 09:15:00 | 1533.75 | 1562.55 | 1572.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-02 09:15:00 | 1524.60 | 1512.88 | 1527.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 09:15:00 | 1524.60 | 1512.88 | 1527.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 1524.60 | 1512.88 | 1527.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:45:00 | 1524.60 | 1512.88 | 1527.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1515.10 | 1513.33 | 1526.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 1525.80 | 1513.33 | 1526.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 12:15:00 | 1526.00 | 1515.86 | 1526.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-04 09:15:00 | 1501.00 | 1515.86 | 1526.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 15:15:00 | 1425.95 | 1454.19 | 1479.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-06 15:15:00 | 1424.00 | 1423.18 | 1447.34 | SL hit (close>ema200) qty=0.50 sl=1423.18 alert=retest2 |

### Cycle 11 — BUY (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 11:15:00 | 1440.90 | 1414.55 | 1412.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 12:15:00 | 1449.50 | 1421.54 | 1415.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 1418.30 | 1428.07 | 1422.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 1418.30 | 1428.07 | 1422.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 1418.30 | 1428.07 | 1422.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:00:00 | 1418.30 | 1428.07 | 1422.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 1420.00 | 1426.45 | 1422.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:15:00 | 1414.35 | 1426.45 | 1422.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 1411.05 | 1423.37 | 1421.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:45:00 | 1410.20 | 1423.37 | 1421.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 13:15:00 | 1408.60 | 1420.42 | 1419.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 13:45:00 | 1406.50 | 1420.42 | 1419.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 15:15:00 | 1411.00 | 1418.47 | 1419.09 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 09:15:00 | 1431.60 | 1421.09 | 1420.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 1464.95 | 1443.03 | 1433.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 09:15:00 | 1487.85 | 1490.67 | 1472.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:15:00 | 1490.00 | 1490.67 | 1472.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 1572.90 | 1575.60 | 1561.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-01 14:45:00 | 1572.00 | 1575.60 | 1561.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 1599.10 | 1605.48 | 1587.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 14:45:00 | 1608.95 | 1605.48 | 1587.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 15:15:00 | 1606.00 | 1605.59 | 1588.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:15:00 | 1614.90 | 1605.59 | 1588.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1619.35 | 1608.34 | 1591.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 11:30:00 | 1625.00 | 1614.46 | 1597.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 15:00:00 | 1631.00 | 1620.16 | 1604.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-15 10:15:00 | 1677.65 | 1699.71 | 1702.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 10:15:00 | 1677.65 | 1699.71 | 1702.09 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 09:15:00 | 1762.20 | 1706.91 | 1703.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 1830.00 | 1780.52 | 1761.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 1832.60 | 1833.42 | 1815.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 1832.60 | 1833.42 | 1815.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1832.60 | 1833.42 | 1815.18 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2024-04-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 11:15:00 | 1794.00 | 1813.41 | 1814.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 12:15:00 | 1785.70 | 1807.87 | 1811.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-29 11:15:00 | 1805.95 | 1785.72 | 1795.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 11:15:00 | 1805.95 | 1785.72 | 1795.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 1805.95 | 1785.72 | 1795.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:00:00 | 1805.95 | 1785.72 | 1795.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 12:15:00 | 1817.00 | 1791.97 | 1797.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-29 12:45:00 | 1824.00 | 1791.97 | 1797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 14:15:00 | 1824.15 | 1804.48 | 1802.92 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 10:15:00 | 1780.35 | 1798.56 | 1800.67 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 11:15:00 | 1817.00 | 1802.25 | 1802.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 12:15:00 | 1819.95 | 1805.79 | 1803.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 15:15:00 | 1830.05 | 1832.23 | 1823.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 15:15:00 | 1830.05 | 1832.23 | 1823.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 1830.05 | 1832.23 | 1823.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:45:00 | 1856.35 | 1836.06 | 1825.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 13:15:00 | 1814.00 | 1829.89 | 1826.25 | SL hit (close<static) qty=1.00 sl=1821.50 alert=retest2 |

### Cycle 20 — SELL (started 2024-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 09:15:00 | 1801.90 | 1819.80 | 1822.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 1771.85 | 1794.71 | 1806.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1814.30 | 1785.66 | 1794.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 1814.30 | 1785.66 | 1794.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1814.30 | 1785.66 | 1794.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:45:00 | 1810.25 | 1785.66 | 1794.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1813.00 | 1791.13 | 1795.98 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 12:15:00 | 1820.75 | 1802.01 | 1800.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 11:15:00 | 1826.00 | 1809.92 | 1805.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 13:15:00 | 1807.05 | 1811.56 | 1806.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-09 13:15:00 | 1807.05 | 1811.56 | 1806.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 1807.05 | 1811.56 | 1806.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:00:00 | 1807.05 | 1811.56 | 1806.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 14:15:00 | 1805.50 | 1810.35 | 1806.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 14:45:00 | 1789.90 | 1810.35 | 1806.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 15:15:00 | 1793.00 | 1806.88 | 1805.53 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1800.10 | 1804.28 | 1804.51 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-10 11:15:00 | 1808.55 | 1805.13 | 1804.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-10 13:15:00 | 1820.00 | 1808.88 | 1806.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 1793.40 | 1807.45 | 1806.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-13 09:15:00 | 1793.40 | 1807.45 | 1806.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1793.40 | 1807.45 | 1806.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:00:00 | 1793.40 | 1807.45 | 1806.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 10:15:00 | 1796.00 | 1805.16 | 1805.74 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1813.10 | 1804.50 | 1804.10 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 14:15:00 | 1802.60 | 1805.22 | 1805.37 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 1815.50 | 1806.93 | 1805.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 1821.25 | 1813.08 | 1810.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1808.10 | 1813.80 | 1811.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1808.10 | 1813.80 | 1811.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1808.10 | 1813.80 | 1811.46 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 1796.50 | 1808.12 | 1809.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 13:15:00 | 1790.40 | 1803.27 | 1806.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 1802.80 | 1786.38 | 1792.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 1802.80 | 1786.38 | 1792.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1802.80 | 1786.38 | 1792.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 12:30:00 | 1781.65 | 1787.68 | 1792.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 15:15:00 | 1780.05 | 1788.43 | 1791.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 1817.00 | 1795.04 | 1793.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1817.00 | 1795.04 | 1793.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 1888.20 | 1814.61 | 1803.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 14:15:00 | 1905.30 | 1917.52 | 1879.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-28 15:00:00 | 1905.30 | 1917.52 | 1879.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1890.15 | 1902.92 | 1883.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 1890.15 | 1902.92 | 1883.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1919.90 | 1906.32 | 1887.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 1936.95 | 1908.31 | 1892.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:00:00 | 1920.10 | 1915.21 | 1900.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-31 09:15:00 | 1941.55 | 1915.35 | 1905.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 10:15:00 | 1869.20 | 1906.24 | 1908.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 10:15:00 | 1869.20 | 1906.24 | 1908.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 13:15:00 | 1859.60 | 1887.87 | 1899.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 1849.45 | 1801.59 | 1828.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 11:15:00 | 1849.45 | 1801.59 | 1828.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1849.45 | 1801.59 | 1828.90 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1895.00 | 1842.48 | 1840.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1949.85 | 1863.95 | 1850.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 12:15:00 | 1919.35 | 1920.64 | 1897.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 1919.35 | 1920.64 | 1897.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 1900.00 | 1916.06 | 1899.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 15:00:00 | 1900.00 | 1916.06 | 1899.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 1900.00 | 1912.85 | 1899.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 09:15:00 | 1911.25 | 1912.85 | 1899.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-10 09:15:00 | 1887.90 | 1907.86 | 1898.22 | SL hit (close<static) qty=1.00 sl=1896.15 alert=retest2 |

### Cycle 32 — SELL (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-10 15:15:00 | 1890.00 | 1894.03 | 1894.23 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-11 09:15:00 | 1918.35 | 1898.90 | 1896.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 11:15:00 | 1928.90 | 1904.83 | 1899.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 14:15:00 | 1942.50 | 1947.46 | 1931.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 15:00:00 | 1942.50 | 1947.46 | 1931.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1933.30 | 1944.63 | 1931.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:15:00 | 1946.85 | 1942.41 | 1932.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 1917.85 | 1936.54 | 1931.87 | SL hit (close<static) qty=1.00 sl=1929.40 alert=retest2 |

### Cycle 34 — SELL (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 14:15:00 | 1899.95 | 1923.37 | 1926.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 11:15:00 | 1893.05 | 1909.28 | 1917.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 14:15:00 | 1925.25 | 1908.92 | 1915.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 14:15:00 | 1925.25 | 1908.92 | 1915.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1925.25 | 1908.92 | 1915.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 1925.25 | 1908.92 | 1915.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 1917.10 | 1910.56 | 1915.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:15:00 | 1925.00 | 1910.56 | 1915.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1932.00 | 1914.85 | 1916.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 09:30:00 | 1931.65 | 1914.85 | 1916.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1982.15 | 1928.31 | 1922.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 11:15:00 | 2084.00 | 1959.45 | 1937.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 15:15:00 | 1973.00 | 1979.06 | 1955.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:15:00 | 1944.80 | 1979.06 | 1955.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1951.25 | 1973.50 | 1955.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:30:00 | 1961.00 | 1973.50 | 1955.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1968.85 | 1972.57 | 1956.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 1976.25 | 1962.80 | 1957.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 1985.10 | 1964.57 | 1960.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 1993.30 | 2001.42 | 2002.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 13:15:00 | 1993.30 | 2001.42 | 2002.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 12:15:00 | 1984.40 | 1992.60 | 1997.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 2016.50 | 1995.85 | 1996.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 2016.50 | 1995.85 | 1996.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 2016.50 | 1995.85 | 1996.89 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 2015.90 | 1999.86 | 1998.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 12:15:00 | 2022.00 | 2007.51 | 2002.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 14:15:00 | 2149.80 | 2152.12 | 2126.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 15:00:00 | 2149.80 | 2152.12 | 2126.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 2259.25 | 2269.62 | 2247.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 2321.10 | 2286.62 | 2274.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 15:15:00 | 2326.00 | 2286.62 | 2274.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 15:15:00 | 2319.30 | 2327.81 | 2318.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 10:00:00 | 2325.30 | 2325.95 | 2318.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 2327.50 | 2329.26 | 2322.45 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 2264.40 | 2312.35 | 2316.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 2264.40 | 2312.35 | 2316.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 2254.75 | 2287.64 | 2302.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 15:15:00 | 2244.10 | 2241.12 | 2261.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 09:15:00 | 2240.70 | 2241.12 | 2261.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 2308.20 | 2246.29 | 2253.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 2308.20 | 2246.29 | 2253.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 15:15:00 | 2265.00 | 2250.03 | 2254.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:15:00 | 2321.45 | 2250.03 | 2254.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 2347.20 | 2269.47 | 2262.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 2375.45 | 2290.66 | 2272.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 12:15:00 | 2417.00 | 2418.95 | 2366.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 12:30:00 | 2422.30 | 2418.95 | 2366.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2430.05 | 2439.22 | 2428.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:45:00 | 2432.30 | 2439.22 | 2428.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 2429.60 | 2437.29 | 2428.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:30:00 | 2432.75 | 2437.29 | 2428.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 2417.25 | 2433.28 | 2427.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 2417.25 | 2433.28 | 2427.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 2406.45 | 2427.92 | 2425.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:45:00 | 2407.30 | 2427.92 | 2425.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 2405.05 | 2423.34 | 2423.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 09:15:00 | 2383.00 | 2411.47 | 2417.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-31 10:15:00 | 2453.60 | 2419.90 | 2420.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 10:15:00 | 2453.60 | 2419.90 | 2420.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 2453.60 | 2419.90 | 2420.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:00:00 | 2453.60 | 2419.90 | 2420.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 2432.10 | 2422.34 | 2421.99 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 12:15:00 | 2399.80 | 2417.83 | 2419.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 2356.70 | 2401.98 | 2412.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 15:15:00 | 2320.00 | 2313.71 | 2338.88 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 09:15:00 | 2270.90 | 2313.71 | 2338.88 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 2342.45 | 2319.46 | 2339.20 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-05 09:15:00 | 2342.45 | 2319.46 | 2339.20 | SL hit (close>ema400) qty=1.00 sl=2339.20 alert=retest1 |

### Cycle 43 — BUY (started 2024-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 10:15:00 | 2407.15 | 2343.53 | 2340.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 09:15:00 | 2412.10 | 2402.65 | 2395.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 2397.10 | 2420.70 | 2410.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 2397.10 | 2420.70 | 2410.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 2397.10 | 2420.70 | 2410.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 2406.80 | 2420.70 | 2410.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 2411.00 | 2418.76 | 2410.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:45:00 | 2430.60 | 2419.29 | 2411.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 12:15:00 | 2293.60 | 2394.15 | 2400.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 2293.60 | 2394.15 | 2400.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 2262.10 | 2352.09 | 2379.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 12:15:00 | 2299.95 | 2295.51 | 2319.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 13:00:00 | 2299.95 | 2295.51 | 2319.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 2287.35 | 2279.39 | 2293.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 11:15:00 | 2277.00 | 2281.10 | 2292.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 13:15:00 | 2279.05 | 2283.49 | 2291.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 2322.90 | 2294.81 | 2293.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 2322.90 | 2294.81 | 2293.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 2337.85 | 2303.42 | 2297.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-27 09:15:00 | 2558.10 | 2572.11 | 2523.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-27 10:00:00 | 2558.10 | 2572.11 | 2523.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 2545.00 | 2556.54 | 2536.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 09:15:00 | 2594.70 | 2556.54 | 2536.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-28 10:00:00 | 2551.95 | 2555.62 | 2537.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 13:15:00 | 2516.00 | 2541.53 | 2536.25 | SL hit (close<static) qty=1.00 sl=2530.50 alert=retest2 |

### Cycle 46 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 2513.00 | 2531.10 | 2532.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 2497.80 | 2524.44 | 2529.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 2510.00 | 2504.51 | 2513.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 2510.00 | 2504.51 | 2513.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 2510.00 | 2504.51 | 2513.98 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 2646.75 | 2532.22 | 2522.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 13:15:00 | 2672.00 | 2608.07 | 2569.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 12:15:00 | 2636.35 | 2641.65 | 2606.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 12:45:00 | 2634.85 | 2641.65 | 2606.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 2612.70 | 2630.20 | 2609.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 2643.00 | 2630.20 | 2609.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2660.00 | 2636.16 | 2614.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 11:00:00 | 2661.35 | 2641.20 | 2618.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 2668.45 | 2651.15 | 2632.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:15:00 | 2725.90 | 2638.47 | 2638.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-17 10:15:00 | 2927.49 | 2835.32 | 2820.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 11:15:00 | 2791.20 | 2823.17 | 2823.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 2787.20 | 2815.97 | 2820.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 15:15:00 | 2809.80 | 2807.44 | 2814.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 09:15:00 | 2822.65 | 2807.44 | 2814.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2766.55 | 2799.26 | 2810.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:30:00 | 2763.80 | 2787.05 | 2803.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 2766.25 | 2780.05 | 2797.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:45:00 | 2757.85 | 2771.96 | 2792.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:15:00 | 2625.61 | 2669.28 | 2697.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 14:15:00 | 2627.94 | 2669.28 | 2697.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 15:15:00 | 2619.96 | 2660.01 | 2690.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-25 13:15:00 | 2665.20 | 2628.57 | 2659.98 | SL hit (close>ema200) qty=0.50 sl=2628.57 alert=retest2 |

### Cycle 49 — BUY (started 2024-09-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 15:15:00 | 2682.00 | 2654.31 | 2651.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 11:15:00 | 2700.00 | 2674.47 | 2662.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 09:15:00 | 2680.05 | 2689.12 | 2675.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 2680.05 | 2689.12 | 2675.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2680.05 | 2689.12 | 2675.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 2664.40 | 2689.12 | 2675.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 2690.00 | 2689.30 | 2676.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 2681.05 | 2689.30 | 2676.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 2677.95 | 2687.03 | 2677.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:00:00 | 2677.95 | 2687.03 | 2677.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 2676.05 | 2684.83 | 2676.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:30:00 | 2680.50 | 2684.83 | 2676.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 2668.55 | 2681.58 | 2676.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:15:00 | 2660.35 | 2681.58 | 2676.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 2653.25 | 2675.91 | 2674.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 2653.25 | 2675.91 | 2674.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 2650.00 | 2670.73 | 2671.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 2599.70 | 2656.52 | 2665.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2561.35 | 2505.48 | 2539.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 2561.35 | 2505.48 | 2539.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 2561.35 | 2505.48 | 2539.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 2555.05 | 2505.48 | 2539.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2605.00 | 2525.39 | 2545.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 2612.70 | 2525.39 | 2545.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 2611.10 | 2561.41 | 2558.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 2689.85 | 2587.10 | 2570.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 2699.40 | 2715.90 | 2684.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 2699.40 | 2715.90 | 2684.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 2705.00 | 2708.69 | 2692.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 2707.90 | 2708.69 | 2692.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 2723.95 | 2708.15 | 2695.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:30:00 | 2732.15 | 2724.63 | 2706.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 12:15:00 | 2727.50 | 2765.92 | 2767.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 2727.50 | 2765.92 | 2767.65 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 15:15:00 | 2796.40 | 2772.81 | 2770.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 09:15:00 | 2830.00 | 2784.25 | 2775.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 09:15:00 | 2883.95 | 2891.19 | 2847.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:45:00 | 2865.10 | 2891.19 | 2847.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 2830.00 | 2868.16 | 2851.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 15:00:00 | 2830.00 | 2868.16 | 2851.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 2835.00 | 2861.53 | 2850.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 2795.80 | 2861.53 | 2850.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 10:15:00 | 2732.75 | 2821.45 | 2833.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 11:15:00 | 2706.80 | 2798.52 | 2821.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 2560.00 | 2545.61 | 2598.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-28 09:30:00 | 2494.00 | 2541.88 | 2591.99 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 2605.75 | 2560.89 | 2592.32 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 2605.75 | 2560.89 | 2592.32 | SL hit (close>ema400) qty=1.00 sl=2592.32 alert=retest1 |

### Cycle 55 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 2596.00 | 2587.70 | 2587.31 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 2580.35 | 2586.23 | 2586.67 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 09:15:00 | 2608.40 | 2588.65 | 2587.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 11:15:00 | 2758.60 | 2629.27 | 2606.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 14:15:00 | 2776.70 | 2781.98 | 2734.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 2776.70 | 2781.98 | 2734.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 2770.00 | 2791.31 | 2767.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 2780.00 | 2791.31 | 2767.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 2838.30 | 2800.71 | 2773.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 2872.00 | 2823.48 | 2799.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:45:00 | 2857.00 | 2835.08 | 2808.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:30:00 | 2854.40 | 2839.01 | 2815.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 12:15:00 | 2772.00 | 2803.67 | 2807.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-11-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 12:15:00 | 2772.00 | 2803.67 | 2807.94 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 11:15:00 | 2823.20 | 2806.90 | 2805.98 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 2785.75 | 2805.15 | 2805.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 2679.55 | 2776.13 | 2791.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 2685.90 | 2639.06 | 2678.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 2685.90 | 2639.06 | 2678.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 2685.90 | 2639.06 | 2678.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:15:00 | 2681.10 | 2639.06 | 2678.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 2681.30 | 2647.51 | 2679.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 09:30:00 | 2658.10 | 2681.75 | 2684.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 10:15:00 | 2697.95 | 2684.99 | 2685.81 | SL hit (close>static) qty=1.00 sl=2693.60 alert=retest2 |

### Cycle 61 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 2700.45 | 2688.08 | 2687.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 2720.85 | 2700.91 | 2694.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 10:15:00 | 2688.55 | 2698.44 | 2693.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 10:15:00 | 2688.55 | 2698.44 | 2693.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2688.55 | 2698.44 | 2693.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 2688.55 | 2698.44 | 2693.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 2684.85 | 2695.72 | 2692.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 2684.85 | 2695.72 | 2692.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 2690.40 | 2694.65 | 2692.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:30:00 | 2683.70 | 2694.65 | 2692.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 2685.00 | 2692.72 | 2691.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 14:00:00 | 2685.00 | 2692.72 | 2691.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 2775.10 | 2719.87 | 2705.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 10:30:00 | 2818.80 | 2741.69 | 2716.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 2935.05 | 2965.15 | 2967.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 12:15:00 | 2935.05 | 2965.15 | 2967.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 14:15:00 | 2930.00 | 2953.15 | 2961.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 2901.90 | 2897.84 | 2924.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 2901.90 | 2897.84 | 2924.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2871.00 | 2882.08 | 2908.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:30:00 | 2870.55 | 2882.08 | 2908.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2878.85 | 2869.63 | 2890.74 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 2901.50 | 2891.25 | 2891.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 2938.90 | 2900.78 | 2895.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 2933.35 | 2935.16 | 2922.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 2933.35 | 2935.16 | 2922.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 2933.35 | 2935.16 | 2922.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 11:00:00 | 2964.80 | 2941.09 | 2926.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:00:00 | 2961.05 | 2949.71 | 2935.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 2876.95 | 2961.29 | 2966.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 2876.95 | 2961.29 | 2966.02 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 2987.10 | 2967.32 | 2965.75 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 2925.80 | 2957.13 | 2961.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 15:15:00 | 2902.50 | 2929.54 | 2943.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 2947.45 | 2933.12 | 2944.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 2947.45 | 2933.12 | 2944.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 2947.45 | 2933.12 | 2944.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:45:00 | 2950.50 | 2933.12 | 2944.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 10:15:00 | 2962.60 | 2939.02 | 2945.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-17 10:45:00 | 2960.60 | 2939.02 | 2945.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 11:15:00 | 3006.55 | 2952.52 | 2951.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 12:15:00 | 3080.10 | 2978.04 | 2963.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 3011.90 | 3012.63 | 2989.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:00:00 | 3011.90 | 3012.63 | 2989.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 2997.95 | 3008.56 | 2991.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 12:45:00 | 2995.65 | 3008.56 | 2991.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 3042.80 | 3015.41 | 2996.20 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 11:15:00 | 2924.90 | 2982.14 | 2987.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 12:15:00 | 2895.60 | 2964.83 | 2979.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 12:15:00 | 2555.80 | 2555.07 | 2615.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 13:00:00 | 2555.80 | 2555.07 | 2615.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 2594.85 | 2563.54 | 2592.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:45:00 | 2595.30 | 2563.54 | 2592.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2602.00 | 2571.23 | 2593.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 13:00:00 | 2602.00 | 2571.23 | 2593.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 13:15:00 | 2615.05 | 2579.99 | 2595.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:00:00 | 2615.05 | 2579.99 | 2595.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 2581.35 | 2583.59 | 2594.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 2606.70 | 2583.59 | 2594.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 2605.80 | 2588.03 | 2595.68 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 12:15:00 | 2649.35 | 2607.86 | 2603.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 2660.80 | 2623.98 | 2611.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 2603.20 | 2621.83 | 2613.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 2603.20 | 2621.83 | 2613.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 2603.20 | 2621.83 | 2613.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:30:00 | 2632.65 | 2621.08 | 2615.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:00:00 | 2633.60 | 2621.08 | 2615.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 2636.85 | 2620.94 | 2616.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 13:15:00 | 2705.00 | 2737.37 | 2737.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 2705.00 | 2737.37 | 2737.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 15:15:00 | 2682.75 | 2723.52 | 2731.21 | Break + close below crossover candle low |

### Cycle 71 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 2788.15 | 2736.45 | 2736.39 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 15:15:00 | 2740.05 | 2752.54 | 2752.79 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 2759.35 | 2753.90 | 2753.38 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 09:15:00 | 2620.15 | 2730.38 | 2744.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 2588.90 | 2640.79 | 2683.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 14:15:00 | 2609.25 | 2594.52 | 2640.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-13 14:15:00 | 2609.25 | 2594.52 | 2640.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 14:15:00 | 2609.25 | 2594.52 | 2640.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-13 15:00:00 | 2609.25 | 2594.52 | 2640.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 2562.20 | 2580.80 | 2607.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 2541.10 | 2577.40 | 2600.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:45:00 | 2550.05 | 2571.14 | 2595.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:30:00 | 2549.00 | 2564.62 | 2580.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:00:00 | 2553.65 | 2562.42 | 2577.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 2580.00 | 2565.94 | 2577.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:00:00 | 2580.00 | 2565.94 | 2577.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 2535.45 | 2559.84 | 2573.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 2511.35 | 2545.71 | 2564.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:30:00 | 2512.85 | 2533.34 | 2548.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:45:00 | 2530.75 | 2538.05 | 2545.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 2414.04 | 2498.00 | 2520.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 2422.55 | 2498.00 | 2520.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 2421.55 | 2498.00 | 2520.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 09:15:00 | 2425.97 | 2498.00 | 2520.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 2385.78 | 2474.88 | 2507.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 2387.21 | 2474.88 | 2507.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-22 10:15:00 | 2404.21 | 2474.88 | 2507.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-23 11:15:00 | 2295.05 | 2362.14 | 2420.02 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 75 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 2340.00 | 2265.31 | 2259.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 2355.45 | 2283.34 | 2267.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 2498.15 | 2504.80 | 2473.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 12:00:00 | 2498.15 | 2504.80 | 2473.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2621.05 | 2544.00 | 2505.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:30:00 | 2568.00 | 2544.00 | 2505.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 2757.35 | 2785.83 | 2735.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:00:00 | 2757.35 | 2785.83 | 2735.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 2769.80 | 2782.62 | 2738.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 2733.70 | 2782.62 | 2738.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2712.95 | 2761.49 | 2746.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 2712.95 | 2761.49 | 2746.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2733.90 | 2755.97 | 2745.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:30:00 | 2704.10 | 2755.97 | 2745.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 2719.70 | 2739.76 | 2739.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:15:00 | 2731.65 | 2739.76 | 2739.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 10:15:00 | 2740.00 | 2758.51 | 2750.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 11:15:00 | 2723.25 | 2743.51 | 2744.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 11:15:00 | 2723.25 | 2743.51 | 2744.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 2682.40 | 2731.29 | 2738.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 2498.40 | 2470.80 | 2518.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 2498.40 | 2470.80 | 2518.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 2560.00 | 2488.64 | 2521.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:00:00 | 2560.00 | 2488.64 | 2521.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2573.35 | 2505.58 | 2526.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 2580.00 | 2505.58 | 2526.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 2556.25 | 2527.52 | 2532.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 11:00:00 | 2556.25 | 2527.52 | 2532.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 11:15:00 | 2554.75 | 2532.96 | 2534.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 12:00:00 | 2554.75 | 2532.96 | 2534.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 2562.85 | 2538.94 | 2537.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 14:15:00 | 2599.85 | 2552.51 | 2543.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 15:15:00 | 2549.00 | 2551.81 | 2544.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 15:15:00 | 2549.00 | 2551.81 | 2544.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 2549.00 | 2551.81 | 2544.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 2542.15 | 2551.81 | 2544.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 2586.85 | 2558.81 | 2548.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 14:00:00 | 2599.30 | 2579.81 | 2562.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 14:30:00 | 2601.95 | 2583.70 | 2565.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 15:00:00 | 2599.25 | 2583.70 | 2565.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 2601.05 | 2573.90 | 2570.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 2551.00 | 2569.32 | 2568.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 2551.00 | 2569.32 | 2568.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-21 10:15:00 | 2536.00 | 2562.66 | 2565.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 2536.00 | 2562.66 | 2565.34 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 11:15:00 | 2596.60 | 2561.05 | 2560.39 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 2531.50 | 2555.80 | 2558.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 2520.15 | 2541.23 | 2550.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 11:15:00 | 2547.55 | 2542.50 | 2550.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 11:15:00 | 2547.55 | 2542.50 | 2550.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 2547.55 | 2542.50 | 2550.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 11:30:00 | 2544.00 | 2542.50 | 2550.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 2550.00 | 2544.00 | 2550.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:45:00 | 2549.90 | 2544.00 | 2550.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 2548.15 | 2544.83 | 2550.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:30:00 | 2550.30 | 2544.83 | 2550.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 2542.40 | 2544.34 | 2549.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 2542.40 | 2544.34 | 2549.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 2540.00 | 2543.47 | 2548.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 2560.40 | 2543.47 | 2548.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 2523.95 | 2539.57 | 2546.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:00:00 | 2503.45 | 2526.29 | 2538.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:45:00 | 2498.95 | 2521.03 | 2534.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 15:00:00 | 2502.85 | 2517.40 | 2531.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 2460.00 | 2518.56 | 2530.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 2378.28 | 2438.69 | 2477.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 15:15:00 | 2377.71 | 2438.69 | 2477.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 2374.00 | 2421.37 | 2466.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 2337.00 | 2399.65 | 2452.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 2380.00 | 2370.74 | 2414.87 | SL hit (close>ema200) qty=0.50 sl=2370.74 alert=retest2 |

### Cycle 81 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 2530.15 | 2444.74 | 2439.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 09:15:00 | 2601.45 | 2517.24 | 2478.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 2707.25 | 2715.37 | 2661.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:00:00 | 2707.25 | 2715.37 | 2661.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2749.95 | 2777.24 | 2733.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 2641.95 | 2777.24 | 2733.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2656.40 | 2753.08 | 2726.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 2656.70 | 2753.08 | 2726.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 2695.40 | 2741.54 | 2723.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:30:00 | 2721.80 | 2732.33 | 2723.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-20 09:15:00 | 2993.98 | 2900.71 | 2876.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 15:15:00 | 2880.00 | 2913.84 | 2914.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 11:15:00 | 2832.20 | 2865.71 | 2883.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 2814.45 | 2803.99 | 2829.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 15:00:00 | 2814.45 | 2803.99 | 2829.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2871.40 | 2817.48 | 2831.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 2873.80 | 2817.48 | 2831.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 2879.65 | 2829.91 | 2835.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 2878.25 | 2829.91 | 2835.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 2877.00 | 2839.33 | 2839.30 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 2739.90 | 2823.33 | 2834.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 11:15:00 | 2733.10 | 2805.28 | 2825.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 09:15:00 | 2790.00 | 2773.57 | 2798.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 10:00:00 | 2790.00 | 2773.57 | 2798.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 2794.60 | 2777.77 | 2798.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 2794.60 | 2777.77 | 2798.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 2792.10 | 2780.64 | 2797.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:45:00 | 2776.90 | 2781.27 | 2796.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:00:00 | 2787.00 | 2782.42 | 2795.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 10:15:00 | 2808.40 | 2792.00 | 2796.28 | SL hit (close>static) qty=1.00 sl=2800.60 alert=retest2 |

### Cycle 85 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 2815.75 | 2800.08 | 2799.42 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 2755.05 | 2795.50 | 2798.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 2730.30 | 2776.28 | 2788.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 2624.05 | 2621.72 | 2678.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 2624.05 | 2621.72 | 2678.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2672.55 | 2640.57 | 2677.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:30:00 | 2695.15 | 2640.57 | 2677.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 2631.00 | 2638.66 | 2673.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 2585.15 | 2638.66 | 2660.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 14:15:00 | 2620.40 | 2610.52 | 2610.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 14:15:00 | 2620.40 | 2610.52 | 2610.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 15:15:00 | 2651.20 | 2618.65 | 2613.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-21 13:15:00 | 2880.80 | 2884.45 | 2832.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-21 13:45:00 | 2880.20 | 2884.45 | 2832.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2925.00 | 2961.87 | 2921.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2922.00 | 2961.87 | 2921.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 2921.50 | 2949.09 | 2922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 12:30:00 | 2921.80 | 2949.09 | 2922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 2922.00 | 2943.68 | 2922.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:30:00 | 2921.60 | 2943.68 | 2922.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 2930.00 | 2940.94 | 2923.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:45:00 | 2930.20 | 2940.94 | 2923.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 2920.00 | 2936.75 | 2923.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 2900.60 | 2936.75 | 2923.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 2964.40 | 2942.28 | 2926.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 2977.90 | 2942.28 | 2926.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 2836.70 | 2920.33 | 2925.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 2836.70 | 2920.33 | 2925.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 2806.40 | 2854.86 | 2885.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2873.10 | 2858.51 | 2884.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 2873.10 | 2858.51 | 2884.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2873.10 | 2858.51 | 2884.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 2868.10 | 2858.51 | 2884.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2830.00 | 2836.38 | 2859.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 14:30:00 | 2808.10 | 2826.48 | 2846.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-06 15:15:00 | 2667.69 | 2730.03 | 2751.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-08 09:15:00 | 2759.70 | 2710.11 | 2725.80 | SL hit (close>ema200) qty=0.50 sl=2710.11 alert=retest2 |

### Cycle 89 — BUY (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 12:15:00 | 2770.20 | 2737.26 | 2735.69 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 2700.00 | 2728.00 | 2731.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 2681.00 | 2718.60 | 2727.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 2680.00 | 2656.54 | 2682.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 15:15:00 | 2680.00 | 2656.54 | 2682.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 2680.00 | 2656.54 | 2682.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 2768.00 | 2656.54 | 2682.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2752.00 | 2675.64 | 2688.81 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2767.10 | 2710.79 | 2703.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2800.00 | 2738.89 | 2718.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 2790.00 | 2794.62 | 2768.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 2851.50 | 2794.62 | 2768.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 2823.50 | 2838.09 | 2817.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 2819.90 | 2838.09 | 2817.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2817.40 | 2831.73 | 2818.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:45:00 | 2818.10 | 2831.73 | 2818.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 2824.20 | 2830.23 | 2818.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 2863.70 | 2830.06 | 2819.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:15:00 | 2837.90 | 2830.83 | 2821.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 2839.80 | 2831.18 | 2822.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 2841.30 | 2831.18 | 2822.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2820.50 | 2838.22 | 2831.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 2790.10 | 2829.03 | 2829.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 13:15:00 | 2790.10 | 2829.03 | 2829.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 2648.20 | 2784.08 | 2808.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 2557.30 | 2556.07 | 2613.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2535.00 | 2556.07 | 2613.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 13:30:00 | 2541.80 | 2555.14 | 2576.91 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-23 15:15:00 | 2537.00 | 2555.76 | 2575.21 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2526.80 | 2546.96 | 2567.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 2508.40 | 2535.04 | 2556.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2408.25 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2414.71 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2410.15 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 15:15:00 | 2382.98 | 2445.22 | 2487.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 2407.70 | 2376.88 | 2401.80 | SL hit (close>ema200) qty=0.50 sl=2376.88 alert=retest1 |

### Cycle 93 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 2452.00 | 2412.80 | 2410.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 2468.50 | 2441.94 | 2429.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2454.20 | 2457.46 | 2445.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:45:00 | 2457.80 | 2457.46 | 2445.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 2457.20 | 2457.41 | 2446.51 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 2440.10 | 2443.72 | 2443.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 15:15:00 | 2435.00 | 2441.38 | 2442.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 2429.80 | 2424.64 | 2430.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 10:30:00 | 2423.50 | 2423.83 | 2429.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:45:00 | 2424.50 | 2423.90 | 2429.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 13:00:00 | 2424.10 | 2423.94 | 2428.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 2413.00 | 2427.02 | 2429.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 15:15:00 | 2413.00 | 2424.21 | 2427.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:15:00 | 2414.00 | 2424.21 | 2427.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2407.00 | 2420.77 | 2426.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 2401.70 | 2413.32 | 2420.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 2400.30 | 2406.18 | 2414.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2302.32 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2303.28 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2302.89 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 09:15:00 | 2292.35 | 2337.35 | 2356.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 2335.00 | 2331.31 | 2346.70 | SL hit (close>ema200) qty=0.50 sl=2331.31 alert=retest2 |

### Cycle 95 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 2380.90 | 2359.44 | 2356.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 2408.00 | 2376.27 | 2365.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 2413.30 | 2423.70 | 2404.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 2413.30 | 2423.70 | 2404.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 2389.70 | 2416.90 | 2403.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 2390.00 | 2416.90 | 2403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 2375.00 | 2408.52 | 2400.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 2375.00 | 2408.52 | 2400.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 2402.50 | 2405.94 | 2400.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 2405.00 | 2404.87 | 2400.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 09:15:00 | 2455.40 | 2488.44 | 2489.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 2455.40 | 2488.44 | 2489.43 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 12:15:00 | 2506.90 | 2491.22 | 2490.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-30 15:15:00 | 2528.00 | 2504.78 | 2497.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 2485.00 | 2500.83 | 2496.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 2491.00 | 2500.83 | 2496.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 2489.90 | 2498.64 | 2495.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:15:00 | 2494.70 | 2498.64 | 2495.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 2482.70 | 2492.04 | 2493.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 2482.70 | 2492.04 | 2493.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 2480.10 | 2488.38 | 2491.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 2456.00 | 2453.46 | 2468.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 2430.40 | 2453.85 | 2457.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2430.20 | 2447.96 | 2454.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 2428.00 | 2434.62 | 2445.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:15:00 | 2429.80 | 2435.21 | 2442.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 2441.70 | 2436.44 | 2441.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 2443.00 | 2436.44 | 2441.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 2433.00 | 2435.76 | 2440.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 2412.10 | 2434.10 | 2439.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 2411.80 | 2376.70 | 2376.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 2411.80 | 2376.70 | 2376.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2421.50 | 2391.88 | 2383.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 2393.50 | 2399.53 | 2390.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 2393.50 | 2399.53 | 2390.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2387.30 | 2397.08 | 2389.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 2401.10 | 2390.74 | 2389.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 2401.00 | 2392.79 | 2390.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 2384.00 | 2390.95 | 2391.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 2384.00 | 2390.95 | 2391.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 15:15:00 | 2373.00 | 2382.21 | 2386.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 2371.50 | 2371.47 | 2379.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 2371.50 | 2371.47 | 2379.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 2374.20 | 2371.46 | 2377.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 2362.30 | 2368.99 | 2375.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 14:15:00 | 2390.70 | 2374.57 | 2375.96 | SL hit (close>static) qty=1.00 sl=2384.70 alert=retest2 |

### Cycle 101 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 2398.00 | 2379.26 | 2377.96 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 2372.50 | 2376.99 | 2377.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 2366.90 | 2374.97 | 2376.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2388.70 | 2375.03 | 2375.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 2388.70 | 2375.03 | 2375.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 2389.00 | 2377.82 | 2376.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 2399.90 | 2388.69 | 2383.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2371.00 | 2385.15 | 2381.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 2365.00 | 2385.15 | 2381.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 2364.40 | 2381.00 | 2380.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 2364.20 | 2381.00 | 2380.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 2371.20 | 2379.04 | 2379.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 2357.80 | 2370.95 | 2374.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2348.00 | 2324.57 | 2341.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2348.00 | 2324.57 | 2341.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2344.00 | 2328.46 | 2341.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:45:00 | 2346.00 | 2328.46 | 2341.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2365.00 | 2335.77 | 2343.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 2365.00 | 2335.77 | 2343.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2348.00 | 2338.21 | 2344.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 2342.80 | 2339.13 | 2343.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 2342.00 | 2338.92 | 2341.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 2356.70 | 2343.99 | 2343.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 2356.70 | 2343.99 | 2343.62 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 2340.10 | 2344.33 | 2344.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 2316.90 | 2334.32 | 2339.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 2339.50 | 2330.31 | 2335.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 2339.50 | 2330.31 | 2335.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 2341.40 | 2332.52 | 2336.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 2341.40 | 2332.52 | 2336.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 2338.50 | 2333.72 | 2336.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 2338.90 | 2333.72 | 2336.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 2339.10 | 2334.80 | 2336.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:45:00 | 2341.80 | 2334.80 | 2336.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 2369.00 | 2341.64 | 2339.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 15:15:00 | 2370.00 | 2347.31 | 2342.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 11:15:00 | 2352.20 | 2352.88 | 2346.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 11:45:00 | 2353.50 | 2352.88 | 2346.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 2386.70 | 2359.50 | 2350.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:30:00 | 2357.80 | 2359.50 | 2350.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 2373.50 | 2376.89 | 2365.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 2387.00 | 2376.89 | 2365.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 2352.50 | 2379.76 | 2372.04 | SL hit (close<static) qty=1.00 sl=2360.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 11:15:00 | 2339.90 | 2365.10 | 2366.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 2303.90 | 2352.86 | 2360.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2446.40 | 2335.73 | 2338.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 2446.40 | 2335.73 | 2338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 2517.00 | 2371.98 | 2354.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 2545.10 | 2406.61 | 2372.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 2431.40 | 2490.62 | 2455.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 2431.40 | 2490.62 | 2455.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 2404.10 | 2473.32 | 2450.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 2415.50 | 2473.32 | 2450.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 11:15:00 | 2395.40 | 2437.77 | 2438.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 12:15:00 | 2356.00 | 2421.41 | 2430.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 2396.90 | 2395.73 | 2411.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 2396.90 | 2395.73 | 2411.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 2395.00 | 2394.07 | 2404.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:15:00 | 2500.50 | 2394.07 | 2404.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 2513.20 | 2417.90 | 2414.45 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 2470.50 | 2481.44 | 2482.72 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 2492.20 | 2483.60 | 2483.58 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 2460.50 | 2480.70 | 2483.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 2457.40 | 2476.04 | 2480.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2440.60 | 2437.32 | 2445.49 | EMA400 retest candle locked (from downside) |

### Cycle 115 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 2486.10 | 2450.08 | 2448.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 2503.40 | 2477.80 | 2465.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 2661.50 | 2669.19 | 2611.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:00:00 | 2661.50 | 2669.19 | 2611.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2605.50 | 2645.19 | 2620.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 2606.90 | 2645.19 | 2620.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2605.10 | 2637.17 | 2619.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:45:00 | 2604.70 | 2637.17 | 2619.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 2600.30 | 2610.13 | 2610.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 2572.00 | 2602.51 | 2607.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 2598.00 | 2592.45 | 2598.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 2589.40 | 2592.45 | 2598.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2600.00 | 2593.96 | 2598.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 2564.60 | 2593.24 | 2594.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 13:15:00 | 2610.50 | 2595.56 | 2595.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 2610.50 | 2595.56 | 2595.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 10:15:00 | 2621.60 | 2602.40 | 2598.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2616.40 | 2626.10 | 2615.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 2616.40 | 2626.10 | 2615.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 2617.20 | 2624.32 | 2616.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 2617.20 | 2624.32 | 2616.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 2603.30 | 2620.11 | 2614.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:00:00 | 2603.30 | 2620.11 | 2614.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2606.80 | 2617.45 | 2614.11 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 15:15:00 | 2595.00 | 2609.69 | 2610.96 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 12:15:00 | 2619.50 | 2612.84 | 2611.94 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 2608.00 | 2611.55 | 2611.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 2588.40 | 2606.99 | 2609.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2607.00 | 2606.99 | 2609.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 2607.00 | 2606.99 | 2609.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2593.90 | 2604.38 | 2607.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:30:00 | 2592.20 | 2600.73 | 2605.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 2592.70 | 2598.21 | 2603.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 2631.30 | 2600.53 | 2603.01 | SL hit (close>static) qty=1.00 sl=2608.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 2632.60 | 2606.95 | 2605.70 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 2603.50 | 2612.57 | 2612.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 10:15:00 | 2590.00 | 2605.05 | 2608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 2520.10 | 2510.19 | 2542.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 2520.10 | 2510.19 | 2542.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 2505.30 | 2517.20 | 2534.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 2536.00 | 2517.20 | 2534.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 2530.00 | 2519.76 | 2533.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 2491.10 | 2516.71 | 2531.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 2492.90 | 2491.49 | 2508.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 2498.10 | 2492.65 | 2500.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 2494.70 | 2492.71 | 2495.50 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 2498.20 | 2491.90 | 2494.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 2496.10 | 2491.90 | 2494.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 2488.70 | 2491.26 | 2494.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 2485.00 | 2491.26 | 2494.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 15:00:00 | 2485.90 | 2489.94 | 2492.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 2631.60 | 2518.12 | 2505.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 09:15:00 | 2631.60 | 2518.12 | 2505.19 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 13:15:00 | 2547.60 | 2561.47 | 2562.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2535.70 | 2548.51 | 2553.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 2556.70 | 2536.51 | 2542.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 2556.60 | 2536.51 | 2542.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 2543.20 | 2537.85 | 2542.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 2525.20 | 2537.85 | 2542.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 2534.50 | 2497.50 | 2508.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 2523.80 | 2499.40 | 2496.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 2523.80 | 2499.40 | 2496.91 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 2486.70 | 2496.15 | 2496.25 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 10:15:00 | 2508.00 | 2496.16 | 2495.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 2520.10 | 2507.98 | 2502.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 2505.90 | 2511.60 | 2506.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 2505.90 | 2511.60 | 2506.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 2513.70 | 2512.02 | 2507.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 2532.20 | 2515.01 | 2509.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 12:30:00 | 2529.90 | 2523.24 | 2517.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 13:15:00 | 2528.30 | 2523.24 | 2517.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 2529.90 | 2527.92 | 2522.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 2555.10 | 2560.01 | 2553.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 2556.00 | 2560.01 | 2553.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 2552.30 | 2558.47 | 2553.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 2509.60 | 2545.47 | 2548.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 2509.60 | 2545.47 | 2548.29 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 14:15:00 | 2551.20 | 2546.72 | 2546.55 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 2524.90 | 2542.35 | 2544.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 2468.00 | 2527.48 | 2537.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 2499.40 | 2487.47 | 2505.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 2503.10 | 2487.47 | 2505.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 2514.30 | 2492.84 | 2506.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 2510.00 | 2492.84 | 2506.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 2509.40 | 2496.15 | 2506.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 2501.00 | 2498.34 | 2506.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 2537.20 | 2506.11 | 2509.67 | SL hit (close>static) qty=1.00 sl=2527.90 alert=retest2 |

### Cycle 131 — BUY (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 10:15:00 | 2552.90 | 2516.43 | 2513.39 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 2514.80 | 2531.10 | 2531.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 11:15:00 | 2508.20 | 2526.52 | 2529.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 2571.30 | 2526.78 | 2526.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 2571.30 | 2526.78 | 2526.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 11:15:00 | 2600.00 | 2541.42 | 2533.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 2621.90 | 2572.89 | 2551.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 2593.60 | 2596.65 | 2578.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 2571.00 | 2591.52 | 2577.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2571.00 | 2591.52 | 2577.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2571.00 | 2591.52 | 2577.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2573.50 | 2587.91 | 2577.43 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 2557.00 | 2570.44 | 2572.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 2539.90 | 2564.33 | 2569.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 2558.50 | 2553.98 | 2561.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 2558.50 | 2553.98 | 2561.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2567.30 | 2556.64 | 2561.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2563.30 | 2556.64 | 2561.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2541.70 | 2553.66 | 2559.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:45:00 | 2540.10 | 2551.09 | 2557.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:15:00 | 2539.40 | 2549.71 | 2556.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 13:45:00 | 2540.00 | 2547.77 | 2554.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:30:00 | 2539.50 | 2545.62 | 2553.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 2530.00 | 2539.19 | 2548.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 10:15:00 | 2528.20 | 2539.19 | 2548.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 2514.40 | 2534.24 | 2545.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 2559.30 | 2526.46 | 2523.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 2559.30 | 2526.46 | 2523.28 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2521.30 | 2531.80 | 2531.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2520.10 | 2529.46 | 2530.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 2500.00 | 2497.45 | 2506.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 10:45:00 | 2499.20 | 2497.45 | 2506.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 2499.00 | 2497.38 | 2505.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 2500.00 | 2497.38 | 2505.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2504.70 | 2499.39 | 2504.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 2504.70 | 2499.39 | 2504.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 2502.10 | 2499.93 | 2504.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 2637.90 | 2499.93 | 2504.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 2624.50 | 2524.85 | 2515.38 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2569.00 | 2590.88 | 2590.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2557.80 | 2584.26 | 2587.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 2598.90 | 2586.67 | 2588.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 2536.60 | 2586.67 | 2588.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:15:00 | 2573.30 | 2553.88 | 2564.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 2569.30 | 2545.28 | 2544.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 2569.30 | 2545.28 | 2544.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 2572.00 | 2550.62 | 2546.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2540.00 | 2572.79 | 2566.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2540.00 | 2572.79 | 2566.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2554.00 | 2569.03 | 2565.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2538.60 | 2569.03 | 2565.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 2546.90 | 2560.57 | 2562.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 2543.00 | 2553.45 | 2557.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 2534.00 | 2530.24 | 2539.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 13:00:00 | 2534.00 | 2530.24 | 2539.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2535.10 | 2531.21 | 2539.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 2536.00 | 2531.21 | 2539.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 2548.90 | 2534.75 | 2540.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 2548.90 | 2534.75 | 2540.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 2527.50 | 2533.30 | 2538.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 2543.30 | 2533.30 | 2538.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2554.70 | 2537.58 | 2540.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 2553.20 | 2537.58 | 2540.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2545.30 | 2539.12 | 2540.79 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 2554.20 | 2542.14 | 2542.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 12:15:00 | 2562.70 | 2546.25 | 2543.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 2571.20 | 2571.68 | 2561.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 2571.20 | 2571.68 | 2561.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 2579.10 | 2576.11 | 2566.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 2570.00 | 2576.11 | 2566.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 2561.60 | 2573.21 | 2566.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 10:45:00 | 2562.30 | 2573.21 | 2566.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 2576.70 | 2573.90 | 2567.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 2575.10 | 2573.90 | 2567.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2571.10 | 2573.34 | 2567.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:45:00 | 2565.00 | 2573.34 | 2567.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2575.20 | 2573.72 | 2568.32 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 2545.10 | 2562.60 | 2564.47 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 2577.20 | 2565.77 | 2565.17 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 12:15:00 | 2553.00 | 2563.07 | 2564.43 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 2582.00 | 2566.86 | 2566.03 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 2553.80 | 2564.26 | 2565.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 12:15:00 | 2542.50 | 2558.01 | 2562.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 2543.00 | 2537.37 | 2547.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:00:00 | 2543.00 | 2537.37 | 2547.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2543.20 | 2538.54 | 2546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2543.20 | 2538.54 | 2546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2553.30 | 2541.91 | 2546.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 2548.90 | 2541.91 | 2546.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2563.40 | 2546.21 | 2548.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 2561.30 | 2546.21 | 2548.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2584.70 | 2553.91 | 2551.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 2617.30 | 2571.50 | 2560.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 2586.00 | 2589.11 | 2575.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 2585.90 | 2589.11 | 2575.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2601.00 | 2595.13 | 2583.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 11:45:00 | 2618.10 | 2600.61 | 2588.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 2600.00 | 2635.01 | 2638.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 2600.00 | 2635.01 | 2638.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 2586.50 | 2610.21 | 2623.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 2492.50 | 2467.26 | 2486.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 2498.40 | 2467.26 | 2486.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 2518.30 | 2477.46 | 2489.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 2518.30 | 2477.46 | 2489.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2521.00 | 2486.17 | 2492.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 2511.00 | 2486.17 | 2492.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 2515.00 | 2491.94 | 2494.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 13:15:00 | 2516.00 | 2496.75 | 2496.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 2516.00 | 2496.75 | 2496.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 2524.90 | 2504.98 | 2500.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 2525.20 | 2526.28 | 2515.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 2505.00 | 2526.28 | 2515.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2557.00 | 2532.43 | 2519.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:45:00 | 2562.70 | 2544.89 | 2528.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 2496.80 | 2527.81 | 2528.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 2496.80 | 2527.81 | 2528.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 2482.10 | 2518.67 | 2524.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2423.50 | 2414.57 | 2448.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 2413.70 | 2414.57 | 2448.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2431.70 | 2416.02 | 2432.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 2431.70 | 2416.02 | 2432.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 2420.60 | 2416.94 | 2431.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:45:00 | 2422.00 | 2416.94 | 2431.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 2374.10 | 2389.32 | 2408.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:45:00 | 2398.00 | 2389.32 | 2408.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 2383.90 | 2370.89 | 2383.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 13:00:00 | 2383.90 | 2370.89 | 2383.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 2340.70 | 2364.85 | 2379.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:30:00 | 2326.90 | 2351.66 | 2369.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 2380.50 | 2346.55 | 2345.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 2380.50 | 2346.55 | 2345.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 2395.00 | 2364.79 | 2355.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2332.50 | 2369.20 | 2363.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2440.00 | 2373.69 | 2369.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:00:00 | 2430.00 | 2419.81 | 2402.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-05 15:15:00 | 2684.00 | 2433.26 | 2427.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 2411.30 | 2421.50 | 2422.69 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 2439.90 | 2425.68 | 2424.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 12:15:00 | 2442.90 | 2431.08 | 2426.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 14:15:00 | 2427.00 | 2431.66 | 2427.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 15:00:00 | 2427.00 | 2431.66 | 2427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 15:15:00 | 2415.70 | 2428.47 | 2426.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:15:00 | 2427.70 | 2428.47 | 2426.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 2437.40 | 2430.25 | 2427.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 2449.20 | 2434.04 | 2429.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 2418.20 | 2448.68 | 2450.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 2418.20 | 2448.68 | 2450.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 2410.00 | 2428.12 | 2438.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 2380.00 | 2376.26 | 2398.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 2380.00 | 2376.26 | 2398.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2367.00 | 2358.71 | 2369.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 2382.60 | 2358.71 | 2369.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2364.10 | 2359.79 | 2369.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 2373.00 | 2359.79 | 2369.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2360.40 | 2359.91 | 2368.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 2363.20 | 2359.91 | 2368.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2338.40 | 2348.03 | 2358.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 2330.00 | 2340.22 | 2352.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 2348.00 | 2320.31 | 2319.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 2348.00 | 2320.31 | 2319.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2380.60 | 2355.54 | 2340.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 2363.40 | 2370.82 | 2357.81 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 14:15:00 | 2342.30 | 2350.53 | 2351.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 2320.00 | 2344.34 | 2348.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 2336.40 | 2329.97 | 2338.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 2336.40 | 2329.97 | 2338.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 2333.70 | 2330.72 | 2338.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2280.00 | 2330.72 | 2338.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-05 10:15:00 | 2166.00 | 2203.02 | 2239.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-09 09:15:00 | 2052.00 | 2102.06 | 2146.59 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 157 — BUY (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 09:15:00 | 2239.30 | 2124.99 | 2114.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 10:15:00 | 2356.40 | 2171.27 | 2136.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 2156.00 | 2188.09 | 2157.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 2156.00 | 2188.09 | 2157.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 2144.50 | 2179.38 | 2156.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:30:00 | 2109.80 | 2164.68 | 2151.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2101.10 | 2151.96 | 2147.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2093.90 | 2151.96 | 2147.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 2103.80 | 2142.33 | 2143.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 2095.70 | 2133.01 | 2138.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2110.00 | 2097.37 | 2115.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2110.00 | 2097.37 | 2115.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2116.00 | 2101.09 | 2115.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 2116.00 | 2101.09 | 2115.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 2125.80 | 2106.04 | 2116.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:30:00 | 2125.90 | 2106.04 | 2116.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 2144.50 | 2113.73 | 2118.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 2139.60 | 2113.73 | 2118.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 2146.30 | 2123.97 | 2122.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 10:15:00 | 2163.60 | 2131.89 | 2126.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2164.90 | 2208.28 | 2184.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 2164.90 | 2208.28 | 2184.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 2157.00 | 2198.02 | 2181.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 2157.00 | 2198.02 | 2181.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 2175.10 | 2187.02 | 2180.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 2174.10 | 2187.02 | 2180.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2250.00 | 2198.35 | 2186.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:45:00 | 2275.10 | 2214.42 | 2195.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:45:00 | 2277.50 | 2226.54 | 2202.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 13:15:00 | 2187.60 | 2210.00 | 2211.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 2187.60 | 2210.00 | 2211.86 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 09:15:00 | 2223.50 | 2212.82 | 2212.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 2337.70 | 2281.53 | 2251.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 2287.40 | 2314.37 | 2288.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:15:00 | 2286.00 | 2314.37 | 2288.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2279.00 | 2307.29 | 2287.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:30:00 | 2278.70 | 2307.29 | 2287.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2279.10 | 2301.65 | 2286.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 2279.10 | 2301.65 | 2286.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 2271.70 | 2293.91 | 2285.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 2271.70 | 2293.91 | 2285.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2258.00 | 2286.73 | 2283.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2258.00 | 2286.73 | 2283.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2026-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 15:15:00 | 2252.00 | 2279.78 | 2280.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2182.50 | 2260.33 | 2271.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2300.00 | 2242.04 | 2254.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 2300.00 | 2242.04 | 2254.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 2296.10 | 2252.85 | 2257.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 2322.10 | 2252.85 | 2257.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2320.60 | 2266.40 | 2263.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 2384.00 | 2355.89 | 2341.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 2370.00 | 2372.03 | 2357.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:15:00 | 2386.00 | 2372.03 | 2357.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 2369.00 | 2371.43 | 2358.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 2369.00 | 2371.43 | 2358.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2385.00 | 2402.90 | 2391.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2393.90 | 2402.90 | 2391.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 2372.80 | 2387.13 | 2387.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 2372.80 | 2387.13 | 2387.69 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 2421.30 | 2393.14 | 2390.28 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 2378.10 | 2389.25 | 2389.98 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2429.80 | 2394.34 | 2391.25 | EMA200 above EMA400 |

### Cycle 168 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 2382.90 | 2391.70 | 2391.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 2367.00 | 2386.76 | 2389.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 2389.60 | 2383.11 | 2386.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 2387.70 | 2383.11 | 2386.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 2404.40 | 2387.37 | 2388.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:00:00 | 2404.40 | 2387.37 | 2388.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 2399.40 | 2389.77 | 2389.43 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 2354.90 | 2383.72 | 2387.00 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 2406.70 | 2392.17 | 2390.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 14:15:00 | 2417.80 | 2397.29 | 2392.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2396.50 | 2399.17 | 2394.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 2396.70 | 2399.17 | 2394.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2390.90 | 2397.51 | 2394.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 2389.50 | 2397.51 | 2394.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 2390.90 | 2396.19 | 2393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:30:00 | 2391.00 | 2396.19 | 2393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 2385.40 | 2394.03 | 2393.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:30:00 | 2385.00 | 2394.03 | 2393.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 2385.20 | 2392.27 | 2392.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 2383.50 | 2390.51 | 2391.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 2324.60 | 2324.40 | 2342.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 14:00:00 | 2324.60 | 2324.40 | 2342.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 2324.80 | 2306.19 | 2318.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 2324.80 | 2306.19 | 2318.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 2320.10 | 2308.97 | 2318.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:00:00 | 2315.20 | 2311.66 | 2318.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 2313.80 | 2312.33 | 2318.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 2344.90 | 2310.93 | 2310.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 2344.90 | 2310.93 | 2310.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 2350.20 | 2330.19 | 2325.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 2356.50 | 2360.24 | 2350.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 2356.50 | 2360.24 | 2350.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 2354.00 | 2359.00 | 2351.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 2344.60 | 2359.00 | 2351.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2363.60 | 2359.92 | 2352.22 | EMA400 retest candle locked (from upside) |

### Cycle 174 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 2326.20 | 2345.59 | 2347.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 09:15:00 | 2325.90 | 2339.99 | 2344.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 12:15:00 | 2336.70 | 2335.44 | 2340.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 12:30:00 | 2338.70 | 2335.44 | 2340.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 13:15:00 | 2330.00 | 2334.35 | 2339.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 13:30:00 | 2332.50 | 2334.35 | 2339.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-11 14:15:00 | 2323.20 | 2332.12 | 2338.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-11 15:00:00 | 2323.20 | 2332.12 | 2338.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-04 09:15:00 | 1300.90 | 2024-01-10 09:15:00 | 1365.95 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-01-04 09:15:00 | 1300.90 | 2024-01-10 12:15:00 | 1341.75 | STOP_HIT | 0.50 | 3.14% |
| BUY | retest2 | 2024-01-09 11:00:00 | 1328.75 | 2024-01-11 09:15:00 | 1461.63 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-23 14:30:00 | 1410.85 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2024-01-24 09:15:00 | 1414.45 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-01-24 10:00:00 | 1412.35 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-01-24 13:00:00 | 1417.40 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-01-25 10:15:00 | 1412.05 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2024-01-25 11:30:00 | 1412.95 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2024-01-25 12:45:00 | 1411.75 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2024-01-25 13:45:00 | 1414.35 | 2024-01-25 14:15:00 | 1452.95 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2024-02-12 13:30:00 | 1509.90 | 2024-02-21 15:15:00 | 1526.00 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-02-28 09:15:00 | 1611.85 | 2024-02-28 11:15:00 | 1557.60 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-03-04 09:15:00 | 1501.00 | 2024-03-05 15:15:00 | 1425.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-04 09:15:00 | 1501.00 | 2024-03-06 15:15:00 | 1424.00 | STOP_HIT | 0.50 | 5.13% |
| BUY | retest2 | 2024-04-03 11:30:00 | 1625.00 | 2024-04-15 10:15:00 | 1677.65 | STOP_HIT | 1.00 | 3.24% |
| BUY | retest2 | 2024-04-03 15:00:00 | 1631.00 | 2024-04-15 10:15:00 | 1677.65 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2024-05-03 09:45:00 | 1856.35 | 2024-05-03 13:15:00 | 1814.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-05-23 12:30:00 | 1781.65 | 2024-05-24 10:15:00 | 1817.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-05-23 15:15:00 | 1780.05 | 2024-05-24 10:15:00 | 1817.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-05-30 09:15:00 | 1936.95 | 2024-06-03 10:15:00 | 1869.20 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2024-05-30 12:00:00 | 1920.10 | 2024-06-03 10:15:00 | 1869.20 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-05-31 09:15:00 | 1941.55 | 2024-06-03 10:15:00 | 1869.20 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2024-06-10 09:15:00 | 1911.25 | 2024-06-10 09:15:00 | 1887.90 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-06-13 10:15:00 | 1946.85 | 2024-06-13 12:15:00 | 1917.85 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-06-20 09:15:00 | 1976.25 | 2024-06-26 13:15:00 | 1993.30 | STOP_HIT | 1.00 | 0.86% |
| BUY | retest2 | 2024-06-21 09:15:00 | 1985.10 | 2024-06-26 13:15:00 | 1993.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-07-12 14:30:00 | 2321.10 | 2024-07-19 09:15:00 | 2264.40 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-07-12 15:15:00 | 2326.00 | 2024-07-19 09:15:00 | 2264.40 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2024-07-16 15:15:00 | 2319.30 | 2024-07-19 09:15:00 | 2264.40 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-07-18 10:00:00 | 2325.30 | 2024-07-19 09:15:00 | 2264.40 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest1 | 2024-08-05 09:15:00 | 2270.90 | 2024-08-05 09:15:00 | 2342.45 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2024-08-05 11:30:00 | 2329.60 | 2024-08-06 10:15:00 | 2407.15 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-08-05 12:15:00 | 2326.15 | 2024-08-06 10:15:00 | 2407.15 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2024-08-05 15:00:00 | 2290.35 | 2024-08-06 10:15:00 | 2407.15 | STOP_HIT | 1.00 | -5.10% |
| BUY | retest2 | 2024-08-13 11:45:00 | 2430.60 | 2024-08-13 12:15:00 | 2293.60 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2024-08-20 11:15:00 | 2277.00 | 2024-08-21 10:15:00 | 2322.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-08-20 13:15:00 | 2279.05 | 2024-08-21 10:15:00 | 2322.90 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-08-28 09:15:00 | 2594.70 | 2024-08-28 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-08-28 10:00:00 | 2551.95 | 2024-08-28 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-04 11:00:00 | 2661.35 | 2024-09-17 10:15:00 | 2927.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-05 09:15:00 | 2668.45 | 2024-09-17 10:15:00 | 2935.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-09-09 09:15:00 | 2725.90 | 2024-09-18 11:15:00 | 2791.20 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2024-09-19 10:30:00 | 2763.80 | 2024-09-24 14:15:00 | 2625.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 12:45:00 | 2766.25 | 2024-09-24 14:15:00 | 2627.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 13:45:00 | 2757.85 | 2024-09-24 15:15:00 | 2619.96 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-19 10:30:00 | 2763.80 | 2024-09-25 13:15:00 | 2665.20 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2024-09-19 12:45:00 | 2766.25 | 2024-09-25 13:15:00 | 2665.20 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2024-09-19 13:45:00 | 2757.85 | 2024-09-25 13:15:00 | 2665.20 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2024-10-14 09:15:00 | 2707.90 | 2024-10-17 12:15:00 | 2727.50 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-10-14 10:45:00 | 2723.95 | 2024-10-17 12:15:00 | 2727.50 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-10-14 14:30:00 | 2732.15 | 2024-10-17 12:15:00 | 2727.50 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2024-10-28 09:30:00 | 2494.00 | 2024-10-28 11:15:00 | 2605.75 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2024-10-28 14:15:00 | 2590.85 | 2024-10-30 12:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-10-28 15:15:00 | 2576.00 | 2024-10-30 12:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-10-29 09:30:00 | 2576.35 | 2024-10-30 12:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-10-30 09:45:00 | 2583.65 | 2024-10-30 12:15:00 | 2596.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-11-07 09:15:00 | 2872.00 | 2024-11-08 12:15:00 | 2772.00 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-11-07 10:45:00 | 2857.00 | 2024-11-08 12:15:00 | 2772.00 | STOP_HIT | 1.00 | -2.98% |
| BUY | retest2 | 2024-11-07 12:30:00 | 2854.40 | 2024-11-08 12:15:00 | 2772.00 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-11-18 09:30:00 | 2658.10 | 2024-11-18 10:15:00 | 2697.95 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-11-21 10:30:00 | 2818.80 | 2024-12-02 12:15:00 | 2935.05 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2024-12-10 11:00:00 | 2964.80 | 2024-12-13 09:15:00 | 2876.95 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-12-10 15:00:00 | 2961.05 | 2024-12-13 09:15:00 | 2876.95 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-12-31 13:30:00 | 2632.65 | 2025-01-06 13:15:00 | 2705.00 | STOP_HIT | 1.00 | 2.75% |
| BUY | retest2 | 2024-12-31 14:00:00 | 2633.60 | 2025-01-06 13:15:00 | 2705.00 | STOP_HIT | 1.00 | 2.71% |
| BUY | retest2 | 2025-01-01 09:15:00 | 2636.85 | 2025-01-06 13:15:00 | 2705.00 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2025-01-15 12:15:00 | 2541.10 | 2025-01-22 09:15:00 | 2414.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 12:45:00 | 2550.05 | 2025-01-22 09:15:00 | 2422.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:30:00 | 2549.00 | 2025-01-22 09:15:00 | 2421.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:00:00 | 2553.65 | 2025-01-22 09:15:00 | 2425.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 09:30:00 | 2511.35 | 2025-01-22 10:15:00 | 2385.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 09:30:00 | 2512.85 | 2025-01-22 10:15:00 | 2387.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:45:00 | 2530.75 | 2025-01-22 10:15:00 | 2404.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-15 12:15:00 | 2541.10 | 2025-01-23 11:15:00 | 2295.05 | TARGET_HIT | 0.50 | 9.68% |
| SELL | retest2 | 2025-01-15 12:45:00 | 2550.05 | 2025-01-23 11:15:00 | 2294.10 | TARGET_HIT | 0.50 | 10.04% |
| SELL | retest2 | 2025-01-16 11:30:00 | 2549.00 | 2025-01-23 11:15:00 | 2298.29 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2025-01-16 13:00:00 | 2553.65 | 2025-01-23 15:15:00 | 2286.99 | TARGET_HIT | 0.50 | 10.44% |
| SELL | retest2 | 2025-01-17 09:30:00 | 2511.35 | 2025-01-24 09:15:00 | 2260.22 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 09:30:00 | 2512.85 | 2025-01-24 09:15:00 | 2261.57 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 09:45:00 | 2530.75 | 2025-01-24 09:15:00 | 2277.68 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-10 13:15:00 | 2731.65 | 2025-02-11 11:15:00 | 2723.25 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-02-11 10:15:00 | 2740.00 | 2025-02-11 11:15:00 | 2723.25 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-02-19 14:00:00 | 2599.30 | 2025-02-21 10:15:00 | 2536.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-02-19 14:30:00 | 2601.95 | 2025-02-21 10:15:00 | 2536.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-02-19 15:00:00 | 2599.25 | 2025-02-21 10:15:00 | 2536.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-02-21 09:15:00 | 2601.05 | 2025-02-21 10:15:00 | 2536.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-27 13:00:00 | 2503.45 | 2025-02-28 15:15:00 | 2378.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 13:45:00 | 2498.95 | 2025-02-28 15:15:00 | 2377.71 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2025-02-27 15:00:00 | 2502.85 | 2025-03-03 09:15:00 | 2374.00 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2025-02-28 09:15:00 | 2460.00 | 2025-03-03 10:15:00 | 2337.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 13:00:00 | 2503.45 | 2025-03-03 15:15:00 | 2380.00 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2025-02-27 13:45:00 | 2498.95 | 2025-03-03 15:15:00 | 2380.00 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2025-02-27 15:00:00 | 2502.85 | 2025-03-03 15:15:00 | 2380.00 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-02-28 09:15:00 | 2460.00 | 2025-03-03 15:15:00 | 2380.00 | STOP_HIT | 0.50 | 3.25% |
| BUY | retest2 | 2025-03-11 13:30:00 | 2721.80 | 2025-03-20 09:15:00 | 2993.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 12:45:00 | 2776.90 | 2025-04-03 10:15:00 | 2808.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-04-02 14:00:00 | 2787.00 | 2025-04-03 10:15:00 | 2808.40 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-04-09 09:15:00 | 2585.15 | 2025-04-15 14:15:00 | 2620.40 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-04-24 10:15:00 | 2977.90 | 2025-04-25 09:15:00 | 2836.70 | STOP_HIT | 1.00 | -4.74% |
| SELL | retest2 | 2025-04-29 14:30:00 | 2808.10 | 2025-05-06 15:15:00 | 2667.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-29 14:30:00 | 2808.10 | 2025-05-08 09:15:00 | 2759.70 | STOP_HIT | 0.50 | 1.72% |
| BUY | retest2 | 2025-05-16 09:15:00 | 2863.70 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-05-16 10:15:00 | 2837.90 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-16 10:45:00 | 2839.80 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-16 11:15:00 | 2841.30 | 2025-05-19 13:15:00 | 2790.10 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest1 | 2025-05-22 11:15:00 | 2535.00 | 2025-05-27 15:15:00 | 2408.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-23 13:30:00 | 2541.80 | 2025-05-27 15:15:00 | 2414.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-23 15:15:00 | 2537.00 | 2025-05-27 15:15:00 | 2410.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2508.40 | 2025-05-27 15:15:00 | 2382.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-05-22 11:15:00 | 2535.00 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest1 | 2025-05-23 13:30:00 | 2541.80 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest1 | 2025-05-23 15:15:00 | 2537.00 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 5.10% |
| SELL | retest2 | 2025-05-26 13:15:00 | 2508.40 | 2025-05-30 09:15:00 | 2407.70 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-06-09 10:30:00 | 2423.50 | 2025-06-16 09:15:00 | 2302.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 11:45:00 | 2424.50 | 2025-06-16 09:15:00 | 2303.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 13:00:00 | 2424.10 | 2025-06-16 09:15:00 | 2302.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 15:15:00 | 2413.00 | 2025-06-16 09:15:00 | 2292.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-09 10:30:00 | 2423.50 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2025-06-09 11:45:00 | 2424.50 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.69% |
| SELL | retest2 | 2025-06-09 13:00:00 | 2424.10 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-06-09 15:15:00 | 2413.00 | 2025-06-16 13:15:00 | 2335.00 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2025-06-10 13:15:00 | 2401.70 | 2025-06-17 10:15:00 | 2380.90 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-06-11 10:30:00 | 2400.30 | 2025-06-17 10:15:00 | 2380.90 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-06-19 15:15:00 | 2405.00 | 2025-06-30 09:15:00 | 2455.40 | STOP_HIT | 1.00 | 2.10% |
| BUY | retest2 | 2025-07-01 11:15:00 | 2494.70 | 2025-07-01 13:15:00 | 2482.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-08 09:45:00 | 2430.40 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.77% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2430.20 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-07-08 15:00:00 | 2428.00 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-07-09 12:15:00 | 2429.80 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-07-10 09:15:00 | 2412.10 | 2025-07-15 11:15:00 | 2411.80 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-17 09:15:00 | 2401.10 | 2025-07-18 10:15:00 | 2384.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-17 10:00:00 | 2401.00 | 2025-07-18 10:15:00 | 2384.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-22 10:45:00 | 2362.30 | 2025-07-22 14:15:00 | 2390.70 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-07-30 09:30:00 | 2342.80 | 2025-07-31 11:15:00 | 2356.70 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-07-31 10:00:00 | 2342.00 | 2025-07-31 11:15:00 | 2356.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-06 12:15:00 | 2387.00 | 2025-08-07 09:15:00 | 2352.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-09-12 09:30:00 | 2564.60 | 2025-09-12 13:15:00 | 2610.50 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-19 10:30:00 | 2592.20 | 2025-09-19 14:15:00 | 2631.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-19 11:30:00 | 2592.70 | 2025-09-19 14:15:00 | 2631.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-30 10:15:00 | 2491.10 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.64% |
| SELL | retest2 | 2025-10-01 09:15:00 | 2492.90 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.56% |
| SELL | retest2 | 2025-10-03 09:15:00 | 2498.10 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-10-06 09:30:00 | 2494.70 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.49% |
| SELL | retest2 | 2025-10-06 13:15:00 | 2485.00 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-10-06 15:00:00 | 2485.90 | 2025-10-07 09:15:00 | 2631.60 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2025-10-14 11:15:00 | 2525.20 | 2025-10-21 14:15:00 | 2523.80 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-10-16 09:45:00 | 2534.50 | 2025-10-21 14:15:00 | 2523.80 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-10-28 14:00:00 | 2532.20 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-29 12:30:00 | 2529.90 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-29 13:15:00 | 2528.30 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-30 10:30:00 | 2529.90 | 2025-11-04 13:15:00 | 2509.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-11-10 13:30:00 | 2501.00 | 2025-11-10 14:15:00 | 2537.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-11-10 15:15:00 | 2490.00 | 2025-11-11 10:15:00 | 2552.90 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-11-20 11:45:00 | 2540.10 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-20 13:15:00 | 2539.40 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-20 13:45:00 | 2540.00 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-20 14:30:00 | 2539.50 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-21 10:15:00 | 2528.20 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-11-21 11:00:00 | 2514.40 | 2025-11-26 09:15:00 | 2559.30 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-09 09:15:00 | 2536.60 | 2025-12-12 11:15:00 | 2569.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-12-10 10:15:00 | 2573.30 | 2025-12-12 11:15:00 | 2569.30 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2026-01-02 11:45:00 | 2618.10 | 2026-01-07 10:15:00 | 2600.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-14 12:15:00 | 2511.00 | 2026-01-14 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-01-14 12:45:00 | 2515.00 | 2026-01-14 13:15:00 | 2516.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-01-19 12:45:00 | 2562.70 | 2026-01-20 11:15:00 | 2496.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2026-01-29 09:30:00 | 2326.90 | 2026-01-30 13:15:00 | 2380.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2440.00 | 2026-02-05 15:15:00 | 2684.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 10:00:00 | 2430.00 | 2026-02-05 15:15:00 | 2673.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-10 11:00:00 | 2449.20 | 2026-02-12 10:15:00 | 2418.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-02-19 13:00:00 | 2330.00 | 2026-02-24 10:15:00 | 2348.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2280.00 | 2026-03-05 10:15:00 | 2166.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2280.00 | 2026-03-09 09:15:00 | 2052.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-03-20 10:45:00 | 2275.10 | 2026-03-23 13:15:00 | 2187.60 | STOP_HIT | 1.00 | -3.85% |
| BUY | retest2 | 2026-03-20 11:45:00 | 2277.50 | 2026-03-23 13:15:00 | 2187.60 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2393.90 | 2026-04-13 14:15:00 | 2372.80 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2026-04-29 13:00:00 | 2315.20 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-04-29 13:30:00 | 2313.80 | 2026-05-04 10:15:00 | 2344.90 | STOP_HIT | 1.00 | -1.34% |
