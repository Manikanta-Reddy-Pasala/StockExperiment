# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 1814.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 168 |
| ALERT2 | 165 |
| ALERT2_SKIP | 87 |
| ALERT3 | 475 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 208 |
| PARTIAL | 7 |
| TARGET_HIT | 0 |
| STOP_HIT | 214 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 220 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 62 / 158
- **Target hits / Stop hits / Partials:** 0 / 213 / 7
- **Avg / median % per leg:** -0.18% / -0.63%
- **Sum % (uncompounded):** -38.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 116 | 30 | 25.9% | 0 | 116 | 0 | -0.35% | -40.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.88% | -1.9% |
| BUY @ 3rd Alert (retest2) | 115 | 30 | 26.1% | 0 | 115 | 0 | -0.34% | -39.0% |
| SELL (all) | 104 | 32 | 30.8% | 0 | 97 | 7 | 0.02% | 2.2% |
| SELL @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 0 | 5 | 0 | -0.01% | -0.0% |
| SELL @ 3rd Alert (retest2) | 99 | 28 | 28.3% | 0 | 92 | 7 | 0.02% | 2.2% |
| retest1 (combined) | 6 | 4 | 66.7% | 0 | 6 | 0 | -0.32% | -1.9% |
| retest2 (combined) | 214 | 58 | 27.1% | 0 | 207 | 7 | -0.17% | -36.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 1409.15 | 1422.12 | 1422.85 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 1425.60 | 1421.19 | 1421.16 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 12:15:00 | 1420.40 | 1421.03 | 1421.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 14:15:00 | 1415.45 | 1419.43 | 1420.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 14:15:00 | 1414.95 | 1412.93 | 1415.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 1414.95 | 1412.93 | 1415.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1414.95 | 1412.93 | 1415.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 15:00:00 | 1414.95 | 1412.93 | 1415.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 1416.50 | 1413.65 | 1415.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 09:45:00 | 1410.75 | 1412.88 | 1415.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 11:15:00 | 1411.00 | 1412.99 | 1415.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-23 09:15:00 | 1424.55 | 1414.92 | 1414.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 1424.55 | 1414.92 | 1414.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-23 10:15:00 | 1437.10 | 1419.35 | 1416.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 1435.50 | 1436.51 | 1429.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 12:00:00 | 1435.50 | 1436.51 | 1429.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 12:15:00 | 1433.95 | 1435.99 | 1429.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-24 13:15:00 | 1432.75 | 1435.99 | 1429.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 13:15:00 | 1433.80 | 1435.56 | 1430.11 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 12:15:00 | 1420.10 | 1426.73 | 1427.45 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1441.55 | 1429.75 | 1428.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-29 09:15:00 | 1457.00 | 1440.97 | 1435.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 13:15:00 | 1446.80 | 1448.20 | 1441.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 13:30:00 | 1444.75 | 1448.20 | 1441.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 1453.30 | 1449.64 | 1443.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 14:15:00 | 1455.10 | 1448.84 | 1445.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:00:00 | 1455.00 | 1455.05 | 1449.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 12:30:00 | 1454.75 | 1453.74 | 1450.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 14:15:00 | 1454.70 | 1453.49 | 1450.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 1450.80 | 1452.95 | 1450.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 1450.80 | 1452.95 | 1450.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 1450.10 | 1452.38 | 1450.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 1447.00 | 1452.38 | 1450.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 1446.50 | 1451.20 | 1449.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-01 10:00:00 | 1446.50 | 1451.20 | 1449.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 1454.10 | 1451.78 | 1450.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 11:45:00 | 1455.45 | 1452.01 | 1450.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 12:45:00 | 1456.50 | 1452.63 | 1450.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 09:15:00 | 1457.00 | 1451.84 | 1451.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 10:30:00 | 1455.25 | 1452.53 | 1451.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 1466.15 | 1475.46 | 1472.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:00:00 | 1466.15 | 1475.46 | 1472.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 1467.75 | 1473.91 | 1471.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 1466.80 | 1473.91 | 1471.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 14:15:00 | 1467.50 | 1472.23 | 1471.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 15:00:00 | 1467.50 | 1472.23 | 1471.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 15:15:00 | 1466.95 | 1471.17 | 1470.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 1470.85 | 1471.17 | 1470.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 11:15:00 | 1470.65 | 1470.81 | 1470.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 11:15:00 | 1470.65 | 1470.81 | 1470.82 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 12:15:00 | 1471.40 | 1470.93 | 1470.87 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 1465.95 | 1469.93 | 1470.42 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 1478.40 | 1469.66 | 1469.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 10:15:00 | 1485.95 | 1472.92 | 1470.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 09:15:00 | 1471.95 | 1479.57 | 1476.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 1471.95 | 1479.57 | 1476.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 1471.95 | 1479.57 | 1476.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:00:00 | 1471.95 | 1479.57 | 1476.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 1472.35 | 1478.13 | 1475.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:30:00 | 1474.30 | 1478.13 | 1475.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 09:15:00 | 1475.00 | 1477.81 | 1476.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 13:30:00 | 1480.85 | 1477.13 | 1476.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 15:15:00 | 1471.55 | 1475.89 | 1476.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 1471.55 | 1475.89 | 1476.04 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 09:15:00 | 1495.90 | 1479.89 | 1477.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 10:15:00 | 1513.80 | 1486.67 | 1481.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 1515.00 | 1526.44 | 1513.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-20 10:00:00 | 1515.00 | 1526.44 | 1513.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 1505.05 | 1522.16 | 1512.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 11:00:00 | 1505.05 | 1522.16 | 1512.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 11:15:00 | 1510.50 | 1519.83 | 1512.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:00:00 | 1513.50 | 1518.56 | 1512.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 13:00:00 | 1513.20 | 1517.38 | 1516.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-22 14:15:00 | 1508.30 | 1515.82 | 1516.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 1508.30 | 1515.82 | 1516.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 15:15:00 | 1507.55 | 1514.16 | 1515.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 1512.25 | 1508.94 | 1512.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 12:15:00 | 1512.25 | 1508.94 | 1512.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 1512.25 | 1508.94 | 1512.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:45:00 | 1515.00 | 1508.94 | 1512.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 1508.20 | 1508.79 | 1511.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-23 15:00:00 | 1500.00 | 1507.03 | 1510.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 09:45:00 | 1498.50 | 1504.58 | 1508.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:00:00 | 1502.85 | 1504.23 | 1508.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 11:30:00 | 1503.00 | 1504.16 | 1507.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 14:15:00 | 1507.90 | 1505.02 | 1507.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 15:00:00 | 1507.90 | 1505.02 | 1507.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 15:15:00 | 1508.50 | 1505.71 | 1507.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:15:00 | 1513.85 | 1505.71 | 1507.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 1507.10 | 1505.99 | 1507.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-27 12:15:00 | 1513.95 | 1508.79 | 1508.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 12:15:00 | 1513.95 | 1508.79 | 1508.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 14:15:00 | 1522.45 | 1512.52 | 1510.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 1520.50 | 1521.61 | 1517.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-28 14:15:00 | 1520.50 | 1521.61 | 1517.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 14:15:00 | 1520.50 | 1521.61 | 1517.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-28 15:00:00 | 1520.50 | 1521.61 | 1517.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 15:15:00 | 1520.90 | 1521.47 | 1517.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-30 09:15:00 | 1531.50 | 1521.47 | 1517.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 10:15:00 | 1585.40 | 1605.48 | 1606.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 10:15:00 | 1585.40 | 1605.48 | 1606.60 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 1625.00 | 1606.35 | 1605.87 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 14:15:00 | 1595.70 | 1606.24 | 1606.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 15:15:00 | 1589.80 | 1602.95 | 1605.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 13:15:00 | 1597.30 | 1595.93 | 1600.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 13:15:00 | 1597.30 | 1595.93 | 1600.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 13:15:00 | 1597.30 | 1595.93 | 1600.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:00:00 | 1597.30 | 1595.93 | 1600.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 14:15:00 | 1591.40 | 1595.02 | 1599.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 14:45:00 | 1597.95 | 1595.02 | 1599.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 1615.00 | 1598.38 | 1600.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 10:00:00 | 1615.00 | 1598.38 | 1600.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2023-07-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 10:15:00 | 1621.85 | 1603.07 | 1602.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 09:15:00 | 1626.80 | 1616.38 | 1613.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-20 09:15:00 | 1631.15 | 1632.54 | 1624.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-20 09:45:00 | 1630.05 | 1632.54 | 1624.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 13:15:00 | 1634.20 | 1633.98 | 1627.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 13:30:00 | 1631.25 | 1633.98 | 1627.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 1630.15 | 1633.41 | 1629.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:00:00 | 1642.50 | 1634.56 | 1630.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 10:30:00 | 1645.05 | 1637.74 | 1633.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-25 13:15:00 | 1644.95 | 1647.23 | 1642.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-26 15:15:00 | 1626.30 | 1643.07 | 1644.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 15:15:00 | 1626.30 | 1643.07 | 1644.37 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 10:15:00 | 1652.00 | 1646.28 | 1645.70 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 11:15:00 | 1640.65 | 1645.16 | 1645.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-27 13:15:00 | 1627.90 | 1640.39 | 1642.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-31 10:15:00 | 1593.70 | 1593.46 | 1608.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-31 10:45:00 | 1596.40 | 1593.46 | 1608.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 1505.70 | 1493.32 | 1504.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 12:45:00 | 1502.80 | 1493.32 | 1504.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 1508.20 | 1496.30 | 1505.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:00:00 | 1508.20 | 1496.30 | 1505.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 14:15:00 | 1508.15 | 1498.67 | 1505.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 14:45:00 | 1509.65 | 1498.67 | 1505.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 15:15:00 | 1509.55 | 1500.84 | 1505.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-08 09:15:00 | 1515.90 | 1500.84 | 1505.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 1501.50 | 1502.62 | 1505.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-08 11:15:00 | 1495.65 | 1502.62 | 1505.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 15:15:00 | 1519.50 | 1508.96 | 1507.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 1519.50 | 1508.96 | 1507.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 12:15:00 | 1526.50 | 1517.07 | 1513.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 09:15:00 | 1516.55 | 1518.51 | 1515.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 1516.55 | 1518.51 | 1515.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 1516.55 | 1518.51 | 1515.19 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 15:15:00 | 1500.00 | 1511.70 | 1513.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 09:15:00 | 1474.95 | 1504.35 | 1509.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 09:15:00 | 1474.65 | 1472.67 | 1482.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 14:15:00 | 1477.50 | 1474.48 | 1479.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 1477.50 | 1474.48 | 1479.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-17 14:30:00 | 1478.90 | 1474.48 | 1479.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 1464.20 | 1472.79 | 1477.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 10:15:00 | 1459.75 | 1472.79 | 1477.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 12:30:00 | 1462.65 | 1468.11 | 1474.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:00:00 | 1461.15 | 1466.69 | 1472.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 11:15:00 | 1479.95 | 1468.63 | 1471.26 | SL hit (close>static) qty=1.00 sl=1478.30 alert=retest2 |

### Cycle 24 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 1475.50 | 1473.11 | 1472.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1489.35 | 1476.98 | 1474.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 12:15:00 | 1477.05 | 1478.83 | 1476.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 12:15:00 | 1477.05 | 1478.83 | 1476.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 12:15:00 | 1477.05 | 1478.83 | 1476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 12:30:00 | 1476.85 | 1478.83 | 1476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 13:15:00 | 1473.50 | 1477.77 | 1476.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 13:45:00 | 1474.90 | 1477.77 | 1476.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 1469.15 | 1476.04 | 1475.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:30:00 | 1468.60 | 1476.04 | 1475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 15:15:00 | 1469.00 | 1474.64 | 1474.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 09:15:00 | 1461.75 | 1472.06 | 1473.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-23 13:15:00 | 1470.65 | 1469.38 | 1471.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 13:15:00 | 1470.65 | 1469.38 | 1471.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 1470.65 | 1469.38 | 1471.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:30:00 | 1469.85 | 1469.38 | 1471.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 1470.75 | 1469.85 | 1471.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-24 09:15:00 | 1488.80 | 1469.85 | 1471.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2023-08-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-24 09:15:00 | 1490.00 | 1473.88 | 1473.13 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 1467.00 | 1473.14 | 1473.54 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-25 09:15:00 | 1489.50 | 1476.42 | 1474.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-25 12:15:00 | 1502.30 | 1486.85 | 1480.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-28 10:15:00 | 1494.10 | 1495.72 | 1488.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-28 11:00:00 | 1494.10 | 1495.72 | 1488.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 1488.00 | 1494.17 | 1488.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:45:00 | 1490.00 | 1494.17 | 1488.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 12:15:00 | 1495.35 | 1494.41 | 1488.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 13:30:00 | 1498.75 | 1494.47 | 1489.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 15:00:00 | 1501.00 | 1495.77 | 1490.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 09:30:00 | 1503.00 | 1505.59 | 1502.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-31 11:15:00 | 1499.00 | 1503.42 | 1502.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 1503.90 | 1503.52 | 1502.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-08-31 13:15:00 | 1494.95 | 1500.98 | 1501.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 1494.95 | 1500.98 | 1501.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 14:15:00 | 1493.55 | 1499.49 | 1500.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 10:15:00 | 1503.00 | 1498.62 | 1499.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 10:15:00 | 1503.00 | 1498.62 | 1499.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 10:15:00 | 1503.00 | 1498.62 | 1499.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 11:00:00 | 1503.00 | 1498.62 | 1499.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 1505.80 | 1500.05 | 1500.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:00:00 | 1505.80 | 1500.05 | 1500.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 12:15:00 | 1507.65 | 1501.57 | 1500.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-01 13:15:00 | 1512.30 | 1503.72 | 1501.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 13:15:00 | 1508.10 | 1509.95 | 1506.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 1508.10 | 1509.95 | 1506.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 1508.10 | 1509.95 | 1506.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 13:45:00 | 1507.90 | 1509.95 | 1506.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 1509.65 | 1509.89 | 1507.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-04 15:15:00 | 1514.00 | 1509.89 | 1507.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:30:00 | 1516.00 | 1510.93 | 1508.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 10:45:00 | 1512.80 | 1510.37 | 1508.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 13:45:00 | 1513.30 | 1510.77 | 1508.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 14:15:00 | 1510.75 | 1510.76 | 1508.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 15:00:00 | 1510.75 | 1510.76 | 1508.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 1511.00 | 1510.81 | 1509.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:15:00 | 1509.20 | 1510.81 | 1509.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1510.60 | 1510.77 | 1509.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 09:30:00 | 1511.20 | 1510.77 | 1509.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 1514.00 | 1511.41 | 1509.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 11:15:00 | 1514.55 | 1511.41 | 1509.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 12:15:00 | 1514.05 | 1511.64 | 1509.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 12:15:00 | 1506.70 | 1510.65 | 1509.67 | SL hit (close<static) qty=1.00 sl=1507.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-09-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 11:15:00 | 1534.75 | 1537.25 | 1537.47 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 13:15:00 | 1555.95 | 1540.46 | 1538.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 10:15:00 | 1560.80 | 1551.25 | 1547.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 1535.90 | 1550.03 | 1548.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 1535.90 | 1550.03 | 1548.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 1535.90 | 1550.03 | 1548.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 15:00:00 | 1535.90 | 1550.03 | 1548.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 15:15:00 | 1538.90 | 1547.81 | 1547.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-18 09:15:00 | 1542.50 | 1547.81 | 1547.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-18 09:15:00 | 1541.25 | 1546.50 | 1546.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 1541.25 | 1546.50 | 1546.77 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 1554.00 | 1548.06 | 1547.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 14:15:00 | 1558.45 | 1551.47 | 1549.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 10:15:00 | 1550.60 | 1552.66 | 1550.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 10:15:00 | 1550.60 | 1552.66 | 1550.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 1550.60 | 1552.66 | 1550.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 11:00:00 | 1550.60 | 1552.66 | 1550.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 11:15:00 | 1557.40 | 1553.61 | 1551.04 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 1526.00 | 1547.87 | 1549.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 11:15:00 | 1522.70 | 1539.67 | 1545.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 09:15:00 | 1537.45 | 1532.90 | 1538.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 09:15:00 | 1537.45 | 1532.90 | 1538.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 1537.45 | 1532.90 | 1538.97 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 09:15:00 | 1582.40 | 1548.36 | 1544.22 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 1557.30 | 1562.84 | 1562.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 11:15:00 | 1554.45 | 1561.17 | 1562.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 09:15:00 | 1551.50 | 1549.47 | 1555.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 10:00:00 | 1551.50 | 1549.47 | 1555.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1542.55 | 1547.33 | 1551.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-29 15:15:00 | 1539.00 | 1547.33 | 1551.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 11:00:00 | 1539.20 | 1542.77 | 1548.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-03 12:15:00 | 1553.45 | 1546.01 | 1548.93 | SL hit (close>static) qty=1.00 sl=1552.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-03 15:15:00 | 1560.70 | 1552.30 | 1551.35 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 1526.55 | 1547.15 | 1549.09 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 09:15:00 | 1576.45 | 1549.94 | 1546.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 10:15:00 | 1594.00 | 1558.75 | 1550.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 1636.15 | 1636.57 | 1623.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 14:00:00 | 1636.15 | 1636.57 | 1623.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 1627.70 | 1634.64 | 1626.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:30:00 | 1631.70 | 1634.64 | 1626.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 1632.40 | 1634.19 | 1626.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:30:00 | 1632.20 | 1634.19 | 1626.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 1628.80 | 1633.11 | 1626.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 12:00:00 | 1628.80 | 1633.11 | 1626.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 12:15:00 | 1634.75 | 1633.44 | 1627.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 13:30:00 | 1636.35 | 1633.81 | 1628.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 14:00:00 | 1635.30 | 1633.81 | 1628.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 12:15:00 | 1635.00 | 1631.80 | 1629.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-16 09:45:00 | 1637.00 | 1639.73 | 1634.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 10:15:00 | 1635.55 | 1638.89 | 1634.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 11:00:00 | 1635.55 | 1638.89 | 1634.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 11:15:00 | 1645.30 | 1640.17 | 1635.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 1667.00 | 1641.51 | 1638.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 11:30:00 | 1651.90 | 1648.25 | 1642.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 10:00:00 | 1649.85 | 1652.69 | 1647.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-18 10:15:00 | 1631.40 | 1648.43 | 1645.77 | SL hit (close<static) qty=1.00 sl=1633.30 alert=retest2 |

### Cycle 41 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 1626.00 | 1641.23 | 1642.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-19 09:15:00 | 1605.35 | 1629.15 | 1636.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 1634.90 | 1627.62 | 1634.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 1634.90 | 1627.62 | 1634.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 1634.90 | 1627.62 | 1634.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:00:00 | 1634.90 | 1627.62 | 1634.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 1634.95 | 1629.09 | 1634.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:45:00 | 1637.85 | 1629.09 | 1634.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 1634.35 | 1630.14 | 1634.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:45:00 | 1634.95 | 1630.14 | 1634.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 1634.15 | 1630.94 | 1634.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 15:00:00 | 1634.15 | 1630.94 | 1634.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 1629.30 | 1630.61 | 1633.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 1624.35 | 1630.61 | 1633.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 1628.90 | 1629.63 | 1632.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 10:30:00 | 1624.45 | 1631.44 | 1632.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 1543.13 | 1563.89 | 1577.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 1547.45 | 1563.89 | 1577.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-30 09:15:00 | 1543.23 | 1563.89 | 1577.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 1562.05 | 1560.45 | 1568.32 | SL hit (close>ema200) qty=0.50 sl=1560.45 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 11:15:00 | 1575.05 | 1569.34 | 1569.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 09:15:00 | 1578.35 | 1572.17 | 1570.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-02 11:15:00 | 1568.55 | 1572.37 | 1571.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 11:15:00 | 1568.55 | 1572.37 | 1571.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 1568.55 | 1572.37 | 1571.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:00:00 | 1568.55 | 1572.37 | 1571.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 1578.95 | 1573.69 | 1571.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:15:00 | 1579.20 | 1573.69 | 1571.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-02 13:45:00 | 1580.05 | 1575.07 | 1572.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 1586.45 | 1574.76 | 1572.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-03 13:15:00 | 1556.00 | 1573.66 | 1573.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 13:15:00 | 1556.00 | 1573.66 | 1573.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 14:15:00 | 1539.40 | 1566.81 | 1570.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 11:15:00 | 1558.45 | 1555.00 | 1562.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-06 12:00:00 | 1558.45 | 1555.00 | 1562.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 12:15:00 | 1561.95 | 1556.39 | 1562.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 12:45:00 | 1560.00 | 1556.39 | 1562.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 1564.55 | 1558.02 | 1562.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 14:00:00 | 1564.55 | 1558.02 | 1562.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 14:15:00 | 1564.80 | 1559.38 | 1563.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 15:00:00 | 1564.80 | 1559.38 | 1563.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 1563.85 | 1560.27 | 1563.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 1593.75 | 1560.27 | 1563.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 1571.50 | 1562.52 | 1563.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 10:15:00 | 1564.50 | 1562.52 | 1563.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 14:15:00 | 1564.05 | 1562.69 | 1563.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 14:15:00 | 1569.40 | 1564.03 | 1563.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 14:15:00 | 1569.40 | 1564.03 | 1563.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 11:15:00 | 1571.10 | 1565.58 | 1564.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 09:15:00 | 1578.65 | 1579.07 | 1574.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 1578.65 | 1579.07 | 1574.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1578.65 | 1579.07 | 1574.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-10 10:45:00 | 1583.60 | 1580.10 | 1575.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 11:00:00 | 1582.50 | 1587.49 | 1583.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 14:45:00 | 1582.15 | 1587.10 | 1584.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 09:30:00 | 1581.65 | 1595.76 | 1591.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-17 09:15:00 | 1613.05 | 1612.16 | 1603.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-11-20 11:15:00 | 1598.15 | 1604.17 | 1604.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-20 11:15:00 | 1598.15 | 1604.17 | 1604.30 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-21 13:15:00 | 1610.75 | 1603.53 | 1602.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 11:15:00 | 1618.95 | 1610.03 | 1606.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 09:15:00 | 1621.80 | 1623.76 | 1618.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1621.80 | 1623.76 | 1618.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1621.80 | 1623.76 | 1618.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 09:30:00 | 1617.75 | 1623.76 | 1618.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 1618.05 | 1622.62 | 1618.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:30:00 | 1623.05 | 1622.62 | 1618.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 1620.05 | 1622.10 | 1618.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 11:30:00 | 1620.00 | 1622.10 | 1618.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 1617.20 | 1621.12 | 1618.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 13:00:00 | 1617.20 | 1621.12 | 1618.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 13:15:00 | 1614.00 | 1619.70 | 1618.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 14:00:00 | 1614.00 | 1619.70 | 1618.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 1617.65 | 1619.29 | 1618.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 09:15:00 | 1624.15 | 1618.43 | 1617.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-13 09:15:00 | 1684.95 | 1702.21 | 1704.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 09:15:00 | 1684.95 | 1702.21 | 1704.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-13 10:15:00 | 1675.65 | 1696.90 | 1701.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 15:15:00 | 1688.00 | 1686.79 | 1693.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-14 09:15:00 | 1712.50 | 1686.79 | 1693.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1725.00 | 1694.43 | 1696.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 10:00:00 | 1725.00 | 1694.43 | 1696.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 1726.30 | 1700.81 | 1699.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 1731.35 | 1706.92 | 1702.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 12:15:00 | 1720.80 | 1721.74 | 1714.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 12:45:00 | 1719.95 | 1721.74 | 1714.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 15:15:00 | 1719.05 | 1726.56 | 1723.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:15:00 | 1718.00 | 1726.56 | 1723.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 1699.90 | 1721.23 | 1720.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 09:45:00 | 1704.35 | 1721.23 | 1720.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2023-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 10:15:00 | 1707.30 | 1718.44 | 1719.73 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 1727.10 | 1718.27 | 1718.10 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 1691.15 | 1713.06 | 1715.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 1681.50 | 1706.75 | 1712.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1678.00 | 1677.96 | 1690.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 10:15:00 | 1686.40 | 1679.65 | 1689.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 1686.40 | 1679.65 | 1689.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:45:00 | 1687.65 | 1679.65 | 1689.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 1679.20 | 1676.60 | 1683.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 10:30:00 | 1669.00 | 1673.24 | 1681.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 10:45:00 | 1671.20 | 1668.93 | 1669.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 11:15:00 | 1670.30 | 1668.93 | 1669.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-28 12:15:00 | 1677.25 | 1671.44 | 1670.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2023-12-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-28 12:15:00 | 1677.25 | 1671.44 | 1670.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 14:15:00 | 1683.80 | 1673.54 | 1671.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 13:15:00 | 1680.10 | 1680.33 | 1676.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-29 14:00:00 | 1680.10 | 1680.33 | 1676.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 1678.65 | 1682.36 | 1679.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:00:00 | 1678.65 | 1682.36 | 1679.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 1678.00 | 1681.49 | 1678.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:45:00 | 1677.05 | 1681.49 | 1678.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 1675.90 | 1680.37 | 1678.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:00:00 | 1675.90 | 1680.37 | 1678.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 1677.80 | 1679.86 | 1678.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:45:00 | 1676.45 | 1679.86 | 1678.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 1672.00 | 1677.51 | 1677.67 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 12:15:00 | 1679.95 | 1677.80 | 1677.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-02 13:15:00 | 1683.00 | 1678.84 | 1678.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 12:15:00 | 1685.55 | 1686.59 | 1683.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-03 12:15:00 | 1685.55 | 1686.59 | 1683.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 12:15:00 | 1685.55 | 1686.59 | 1683.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 12:45:00 | 1682.20 | 1686.59 | 1683.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 13:15:00 | 1684.30 | 1686.14 | 1683.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 13:30:00 | 1682.90 | 1686.14 | 1683.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 14:15:00 | 1677.10 | 1684.33 | 1682.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-03 15:00:00 | 1677.10 | 1684.33 | 1682.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 15:15:00 | 1679.00 | 1683.26 | 1682.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 09:15:00 | 1711.60 | 1683.26 | 1682.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-09 14:15:00 | 1682.80 | 1699.39 | 1699.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 14:15:00 | 1682.80 | 1699.39 | 1699.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-10 10:15:00 | 1670.05 | 1687.52 | 1693.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-11 09:15:00 | 1693.70 | 1679.10 | 1685.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 09:15:00 | 1693.70 | 1679.10 | 1685.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 1693.70 | 1679.10 | 1685.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-11 09:30:00 | 1698.50 | 1679.10 | 1685.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 1679.60 | 1679.20 | 1684.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-11 11:30:00 | 1677.10 | 1678.72 | 1684.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 10:15:00 | 1593.24 | 1619.65 | 1633.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 1595.90 | 1585.93 | 1597.74 | SL hit (close>ema200) qty=0.50 sl=1585.93 alert=retest2 |

### Cycle 56 — BUY (started 2024-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 10:15:00 | 1603.70 | 1593.73 | 1593.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 11:15:00 | 1612.95 | 1597.58 | 1595.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-29 10:15:00 | 1625.00 | 1625.49 | 1616.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-29 11:00:00 | 1625.00 | 1625.49 | 1616.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 09:15:00 | 1612.60 | 1627.61 | 1622.24 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 1609.95 | 1618.63 | 1619.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 13:15:00 | 1593.50 | 1613.60 | 1616.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 10:15:00 | 1609.45 | 1605.33 | 1610.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 10:15:00 | 1609.45 | 1605.33 | 1610.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 1609.45 | 1605.33 | 1610.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:45:00 | 1610.25 | 1605.33 | 1610.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 11:15:00 | 1615.55 | 1607.38 | 1611.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 12:00:00 | 1615.55 | 1607.38 | 1611.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 12:15:00 | 1620.20 | 1609.94 | 1612.21 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 1628.85 | 1615.19 | 1614.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-02 09:15:00 | 1642.05 | 1625.50 | 1620.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-05 09:15:00 | 1645.15 | 1645.53 | 1635.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-05 10:15:00 | 1640.90 | 1645.53 | 1635.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 10:15:00 | 1638.00 | 1644.02 | 1636.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:45:00 | 1636.95 | 1644.02 | 1636.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 1629.00 | 1641.02 | 1635.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:00:00 | 1629.00 | 1641.02 | 1635.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 1625.20 | 1637.86 | 1634.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-05 12:30:00 | 1619.65 | 1637.86 | 1634.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 1613.35 | 1629.63 | 1631.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 11:15:00 | 1596.65 | 1616.63 | 1624.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-07 09:15:00 | 1617.25 | 1606.23 | 1614.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 09:15:00 | 1617.25 | 1606.23 | 1614.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 09:15:00 | 1617.25 | 1606.23 | 1614.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-07 09:30:00 | 1619.00 | 1606.23 | 1614.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 10:15:00 | 1600.10 | 1605.01 | 1613.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 11:15:00 | 1597.05 | 1605.01 | 1613.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 14:45:00 | 1594.70 | 1601.75 | 1608.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 15:15:00 | 1594.85 | 1601.75 | 1608.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-14 14:15:00 | 1578.95 | 1572.03 | 1571.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 1578.95 | 1572.03 | 1571.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 1583.00 | 1574.23 | 1572.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 10:15:00 | 1571.65 | 1574.41 | 1573.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 1571.65 | 1574.41 | 1573.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 1571.65 | 1574.41 | 1573.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 1570.95 | 1574.41 | 1573.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 1571.35 | 1573.80 | 1573.03 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 13:15:00 | 1567.10 | 1571.63 | 1572.13 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1581.35 | 1573.12 | 1572.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 1589.10 | 1576.85 | 1574.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 12:15:00 | 1609.05 | 1609.19 | 1598.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-20 12:45:00 | 1608.45 | 1609.19 | 1598.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 1602.35 | 1607.77 | 1599.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:00:00 | 1602.35 | 1607.77 | 1599.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 1596.40 | 1605.79 | 1601.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 12:00:00 | 1596.40 | 1605.79 | 1601.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 1598.50 | 1604.33 | 1601.03 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 1588.55 | 1598.92 | 1599.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-22 09:15:00 | 1574.85 | 1592.37 | 1595.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 1592.80 | 1583.19 | 1588.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 1592.80 | 1583.19 | 1588.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1592.80 | 1583.19 | 1588.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1592.80 | 1583.19 | 1588.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1593.85 | 1585.33 | 1589.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1600.85 | 1585.33 | 1589.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 10:15:00 | 1600.75 | 1592.20 | 1592.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 1616.00 | 1598.28 | 1594.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 15:15:00 | 1611.95 | 1615.38 | 1608.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-27 09:15:00 | 1615.55 | 1615.38 | 1608.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 09:15:00 | 1606.05 | 1613.51 | 1608.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 1606.05 | 1613.51 | 1608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 1616.70 | 1614.15 | 1609.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 12:30:00 | 1620.60 | 1614.56 | 1610.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 13:15:00 | 1601.15 | 1611.88 | 1609.63 | SL hit (close<static) qty=1.00 sl=1605.65 alert=retest2 |

### Cycle 65 — SELL (started 2024-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 15:15:00 | 1595.00 | 1606.87 | 1607.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1588.85 | 1600.56 | 1604.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 1594.40 | 1591.12 | 1595.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 15:00:00 | 1594.40 | 1591.12 | 1595.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 1593.85 | 1591.67 | 1595.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 1601.00 | 1591.67 | 1595.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1614.85 | 1596.31 | 1597.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:00:00 | 1614.85 | 1596.31 | 1597.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 1612.05 | 1599.45 | 1598.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 15:15:00 | 1622.70 | 1614.10 | 1609.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 10:15:00 | 1608.90 | 1613.84 | 1610.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 10:15:00 | 1608.90 | 1613.84 | 1610.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 10:15:00 | 1608.90 | 1613.84 | 1610.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 10:45:00 | 1610.00 | 1613.84 | 1610.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1604.55 | 1611.98 | 1609.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 1604.55 | 1611.98 | 1609.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1598.40 | 1609.26 | 1608.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:45:00 | 1606.15 | 1609.26 | 1608.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 1573.30 | 1602.07 | 1605.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 14:15:00 | 1554.00 | 1592.46 | 1600.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 1556.00 | 1555.94 | 1572.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:15:00 | 1584.55 | 1555.94 | 1572.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1581.85 | 1561.13 | 1573.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:30:00 | 1581.80 | 1561.13 | 1573.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 1574.60 | 1563.82 | 1573.31 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 09:15:00 | 1600.20 | 1581.10 | 1578.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 11:15:00 | 1612.55 | 1590.62 | 1583.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 15:15:00 | 1595.00 | 1596.66 | 1589.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-12 09:15:00 | 1583.45 | 1596.66 | 1589.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 1601.35 | 1597.60 | 1590.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 09:30:00 | 1606.45 | 1597.60 | 1590.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1585.00 | 1595.08 | 1589.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:45:00 | 1581.35 | 1595.08 | 1589.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 1585.65 | 1593.19 | 1589.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:45:00 | 1582.65 | 1593.19 | 1589.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 1596.95 | 1593.94 | 1590.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 12:30:00 | 1589.10 | 1593.94 | 1590.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 1593.00 | 1593.76 | 1590.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:00:00 | 1593.00 | 1593.76 | 1590.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 1593.40 | 1593.68 | 1590.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 15:15:00 | 1590.00 | 1593.68 | 1590.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 15:15:00 | 1590.00 | 1592.95 | 1590.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 09:15:00 | 1586.95 | 1592.95 | 1590.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1593.90 | 1593.14 | 1590.95 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 1577.25 | 1588.36 | 1589.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 09:15:00 | 1567.90 | 1577.35 | 1582.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 09:15:00 | 1586.60 | 1572.51 | 1576.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 09:15:00 | 1586.60 | 1572.51 | 1576.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 1586.60 | 1572.51 | 1576.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 10:00:00 | 1586.60 | 1572.51 | 1576.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 10:15:00 | 1567.75 | 1571.56 | 1576.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 11:45:00 | 1562.35 | 1570.16 | 1575.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 13:15:00 | 1564.10 | 1569.52 | 1574.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 14:15:00 | 1562.85 | 1569.02 | 1573.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 15:15:00 | 1562.20 | 1569.62 | 1573.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1562.20 | 1568.14 | 1572.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 1558.75 | 1572.54 | 1572.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:45:00 | 1559.20 | 1570.45 | 1571.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 13:45:00 | 1560.60 | 1567.00 | 1569.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 14:15:00 | 1581.80 | 1569.96 | 1570.59 | SL hit (close>static) qty=1.00 sl=1573.25 alert=retest2 |

### Cycle 70 — BUY (started 2024-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 15:15:00 | 1580.00 | 1571.97 | 1571.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 11:15:00 | 1583.90 | 1574.61 | 1572.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 15:15:00 | 1596.10 | 1596.39 | 1588.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-22 09:15:00 | 1595.80 | 1596.39 | 1588.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 1590.00 | 1594.74 | 1589.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 12:00:00 | 1590.00 | 1594.74 | 1589.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 1594.00 | 1594.60 | 1590.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-22 14:00:00 | 1594.40 | 1594.56 | 1590.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-22 14:15:00 | 1585.95 | 1592.84 | 1590.03 | SL hit (close<static) qty=1.00 sl=1586.35 alert=retest2 |

### Cycle 71 — SELL (started 2024-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 13:15:00 | 1587.00 | 1594.98 | 1595.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 14:15:00 | 1580.20 | 1592.02 | 1594.33 | Break + close below crossover candle low |

### Cycle 72 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 1636.75 | 1599.84 | 1597.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 10:15:00 | 1657.15 | 1611.31 | 1602.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 12:15:00 | 1642.20 | 1644.69 | 1630.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-01 13:00:00 | 1642.20 | 1644.69 | 1630.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1628.00 | 1641.39 | 1633.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 10:00:00 | 1628.00 | 1641.39 | 1633.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 10:15:00 | 1655.80 | 1644.27 | 1635.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 11:15:00 | 1657.80 | 1644.27 | 1635.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 12:15:00 | 1659.60 | 1646.82 | 1637.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 14:30:00 | 1660.80 | 1651.20 | 1641.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 13:30:00 | 1657.90 | 1646.19 | 1643.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1694.35 | 1701.79 | 1695.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 1696.70 | 1701.79 | 1695.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 1699.00 | 1701.23 | 1696.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-15 09:15:00 | 1678.00 | 1694.73 | 1695.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 1678.00 | 1694.73 | 1695.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 1657.50 | 1677.69 | 1686.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-19 13:15:00 | 1612.60 | 1600.90 | 1617.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-19 14:00:00 | 1612.60 | 1600.90 | 1617.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 14:15:00 | 1619.30 | 1604.58 | 1617.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-19 14:30:00 | 1623.30 | 1604.58 | 1617.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 15:15:00 | 1618.60 | 1607.38 | 1618.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:15:00 | 1631.10 | 1607.38 | 1618.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1629.05 | 1611.72 | 1619.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:00:00 | 1620.00 | 1619.24 | 1621.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:30:00 | 1620.00 | 1620.19 | 1621.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 1631.30 | 1622.41 | 1622.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 1631.30 | 1622.41 | 1622.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 1642.55 | 1627.80 | 1624.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 14:15:00 | 1622.80 | 1630.48 | 1627.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 14:15:00 | 1622.80 | 1630.48 | 1627.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 14:15:00 | 1622.80 | 1630.48 | 1627.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 15:00:00 | 1622.80 | 1630.48 | 1627.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 1621.30 | 1628.65 | 1627.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 1626.30 | 1628.65 | 1627.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:45:00 | 1627.25 | 1629.19 | 1627.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 09:30:00 | 1626.00 | 1632.83 | 1631.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:00:00 | 1628.00 | 1632.83 | 1631.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1629.10 | 1632.08 | 1630.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 1624.35 | 1632.08 | 1630.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1636.95 | 1633.06 | 1631.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:15:00 | 1639.10 | 1633.06 | 1631.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1604.60 | 1638.08 | 1636.03 | SL hit (close<static) qty=1.00 sl=1619.15 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 10:15:00 | 1600.80 | 1630.62 | 1632.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-26 11:15:00 | 1597.85 | 1624.07 | 1629.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-30 09:15:00 | 1605.25 | 1599.05 | 1606.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 1605.25 | 1599.05 | 1606.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1605.25 | 1599.05 | 1606.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 09:30:00 | 1605.45 | 1599.05 | 1606.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 1608.80 | 1601.00 | 1607.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:45:00 | 1613.00 | 1601.00 | 1607.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 1619.40 | 1604.68 | 1608.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:00:00 | 1619.40 | 1604.68 | 1608.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1619.95 | 1607.73 | 1609.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-30 12:45:00 | 1624.00 | 1607.73 | 1609.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 13:15:00 | 1624.10 | 1611.01 | 1610.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-03 09:15:00 | 1660.25 | 1626.26 | 1619.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-03 12:15:00 | 1624.80 | 1630.89 | 1623.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 12:15:00 | 1624.80 | 1630.89 | 1623.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 1624.80 | 1630.89 | 1623.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:00:00 | 1624.80 | 1630.89 | 1623.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 13:15:00 | 1625.00 | 1629.72 | 1623.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 13:30:00 | 1623.30 | 1629.72 | 1623.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 1628.45 | 1629.46 | 1624.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-03 14:45:00 | 1624.10 | 1629.46 | 1624.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 15:15:00 | 1623.00 | 1628.17 | 1624.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 09:15:00 | 1632.15 | 1628.17 | 1624.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 09:15:00 | 1625.00 | 1627.54 | 1624.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 10:00:00 | 1625.00 | 1627.54 | 1624.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1624.00 | 1626.83 | 1624.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:15:00 | 1624.00 | 1626.83 | 1624.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 1624.80 | 1626.42 | 1624.30 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 13:15:00 | 1613.00 | 1621.59 | 1622.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 09:15:00 | 1600.35 | 1614.79 | 1618.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 12:15:00 | 1616.50 | 1614.60 | 1617.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 12:15:00 | 1616.50 | 1614.60 | 1617.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1616.50 | 1614.60 | 1617.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:30:00 | 1621.00 | 1614.60 | 1617.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 1611.50 | 1613.98 | 1617.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:30:00 | 1614.90 | 1613.98 | 1617.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 1618.50 | 1614.88 | 1617.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:00:00 | 1618.50 | 1614.88 | 1617.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1612.45 | 1614.40 | 1616.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 1606.95 | 1614.40 | 1616.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:30:00 | 1610.45 | 1613.17 | 1615.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:00:00 | 1610.25 | 1613.47 | 1615.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 1598.75 | 1578.80 | 1577.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 1598.75 | 1578.80 | 1577.09 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 1579.45 | 1583.09 | 1583.35 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 1586.90 | 1583.36 | 1583.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 1589.80 | 1585.20 | 1584.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 10:15:00 | 1584.55 | 1585.07 | 1584.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 10:15:00 | 1584.55 | 1585.07 | 1584.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 1584.55 | 1585.07 | 1584.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 1584.70 | 1585.07 | 1584.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 1585.60 | 1585.18 | 1584.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 11:45:00 | 1588.10 | 1585.18 | 1584.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 1608.45 | 1589.83 | 1586.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 1616.25 | 1598.99 | 1592.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 10:30:00 | 1612.40 | 1602.23 | 1594.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 1593.45 | 1598.32 | 1598.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1593.45 | 1598.32 | 1598.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 09:15:00 | 1588.90 | 1596.28 | 1597.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 15:15:00 | 1531.00 | 1530.65 | 1543.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 09:15:00 | 1570.15 | 1530.65 | 1543.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1575.60 | 1539.64 | 1546.77 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 1576.65 | 1552.70 | 1551.82 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1496.90 | 1550.82 | 1553.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1429.90 | 1526.64 | 1542.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1506.55 | 1502.39 | 1520.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 11:00:00 | 1506.55 | 1502.39 | 1520.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1513.60 | 1504.63 | 1519.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 1513.60 | 1504.63 | 1519.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 1518.20 | 1507.35 | 1519.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 1518.20 | 1507.35 | 1519.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1519.65 | 1509.81 | 1519.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:00:00 | 1519.65 | 1509.81 | 1519.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1518.95 | 1511.64 | 1519.50 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 1531.00 | 1523.25 | 1522.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1552.00 | 1529.00 | 1525.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 1568.80 | 1571.38 | 1557.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 14:45:00 | 1566.80 | 1571.38 | 1557.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1567.35 | 1569.07 | 1558.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 1564.00 | 1569.07 | 1558.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1578.40 | 1571.59 | 1565.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 1568.60 | 1571.59 | 1565.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1587.10 | 1586.33 | 1580.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:45:00 | 1585.85 | 1586.33 | 1580.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1597.45 | 1590.09 | 1585.49 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2024-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 10:15:00 | 1580.50 | 1587.58 | 1588.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 1577.00 | 1584.57 | 1586.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 1585.95 | 1584.84 | 1586.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 14:15:00 | 1585.95 | 1584.84 | 1586.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1585.95 | 1584.84 | 1586.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:15:00 | 1587.50 | 1584.84 | 1586.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1587.50 | 1585.37 | 1586.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 1593.55 | 1585.37 | 1586.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1585.00 | 1585.30 | 1586.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:30:00 | 1582.95 | 1585.24 | 1586.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 12:30:00 | 1583.55 | 1585.69 | 1586.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 13:15:00 | 1593.60 | 1587.27 | 1586.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 1593.60 | 1587.27 | 1586.97 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1572.80 | 1584.38 | 1585.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 1570.35 | 1580.86 | 1583.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 1580.95 | 1579.97 | 1582.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 11:45:00 | 1580.05 | 1579.97 | 1582.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 1581.55 | 1580.29 | 1582.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 09:45:00 | 1578.35 | 1582.13 | 1582.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1576.75 | 1582.13 | 1582.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 12:15:00 | 1589.05 | 1581.50 | 1582.28 | SL hit (close>static) qty=1.00 sl=1585.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 13:15:00 | 1604.75 | 1586.15 | 1584.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 09:15:00 | 1609.70 | 1600.84 | 1595.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 1599.95 | 1602.95 | 1597.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 13:00:00 | 1599.95 | 1602.95 | 1597.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1601.65 | 1602.69 | 1598.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 13:30:00 | 1597.00 | 1602.69 | 1598.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1604.45 | 1603.04 | 1598.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1598.90 | 1603.04 | 1598.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 1599.75 | 1602.38 | 1598.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 1608.55 | 1602.38 | 1598.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1611.90 | 1604.29 | 1600.00 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 1586.55 | 1598.56 | 1598.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 09:15:00 | 1584.00 | 1594.43 | 1596.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 14:15:00 | 1580.35 | 1576.34 | 1582.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-02 15:00:00 | 1580.35 | 1576.34 | 1582.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 1582.00 | 1577.47 | 1582.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 1586.65 | 1577.47 | 1582.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1585.90 | 1579.15 | 1582.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:45:00 | 1589.05 | 1579.15 | 1582.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1582.50 | 1579.82 | 1582.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 1581.25 | 1579.82 | 1582.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 12:15:00 | 1591.45 | 1583.70 | 1584.02 | SL hit (close>static) qty=1.00 sl=1589.60 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 1591.70 | 1585.30 | 1584.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1594.95 | 1587.23 | 1585.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 1588.60 | 1592.41 | 1589.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 13:15:00 | 1588.60 | 1592.41 | 1589.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1588.60 | 1592.41 | 1589.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 1587.55 | 1592.41 | 1589.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1584.50 | 1590.83 | 1589.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 1584.50 | 1590.83 | 1589.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1583.60 | 1589.38 | 1588.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 1576.30 | 1589.38 | 1588.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-07-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 09:15:00 | 1576.85 | 1586.88 | 1587.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 13:15:00 | 1572.80 | 1580.11 | 1583.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 1580.20 | 1580.12 | 1583.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 1580.20 | 1580.12 | 1583.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1574.10 | 1571.55 | 1575.80 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 1582.15 | 1577.43 | 1577.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 1586.95 | 1580.60 | 1578.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 13:15:00 | 1575.00 | 1580.08 | 1579.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 13:15:00 | 1575.00 | 1580.08 | 1579.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 1575.00 | 1580.08 | 1579.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:00:00 | 1575.00 | 1580.08 | 1579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 1581.95 | 1580.45 | 1579.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 1583.00 | 1580.45 | 1579.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:45:00 | 1585.45 | 1581.66 | 1580.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 1626.00 | 1631.16 | 1631.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 1626.00 | 1631.16 | 1631.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 1602.65 | 1625.46 | 1629.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 1579.25 | 1577.95 | 1588.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 10:45:00 | 1578.45 | 1577.95 | 1588.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 1586.30 | 1579.83 | 1587.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:45:00 | 1585.85 | 1579.83 | 1587.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 1584.15 | 1580.70 | 1587.01 | EMA400 retest candle locked (from downside) |

### Cycle 94 — BUY (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 11:15:00 | 1608.90 | 1592.48 | 1590.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 14:15:00 | 1618.60 | 1601.54 | 1595.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 1643.05 | 1646.79 | 1636.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 10:00:00 | 1643.05 | 1646.79 | 1636.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 1644.40 | 1647.04 | 1639.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:30:00 | 1640.00 | 1647.04 | 1639.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1635.50 | 1644.74 | 1639.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1635.50 | 1644.74 | 1639.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1632.65 | 1642.32 | 1638.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1615.10 | 1642.32 | 1638.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1613.35 | 1636.52 | 1636.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1594.50 | 1620.22 | 1627.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1586.20 | 1583.58 | 1601.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 1568.35 | 1581.27 | 1598.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:00:00 | 1572.05 | 1581.27 | 1598.53 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:45:00 | 1569.80 | 1578.78 | 1593.06 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-07 11:15:00 | 1569.65 | 1573.15 | 1585.28 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1557.00 | 1556.83 | 1561.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 1553.50 | 1556.83 | 1561.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1555.00 | 1556.47 | 1561.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 1565.40 | 1558.25 | 1561.66 | SL hit (close>ema400) qty=1.00 sl=1561.66 alert=retest1 |

### Cycle 96 — BUY (started 2024-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 10:15:00 | 1549.70 | 1544.94 | 1544.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 1571.65 | 1553.70 | 1549.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 1711.50 | 1711.50 | 1693.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 1711.50 | 1711.50 | 1693.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 1844.50 | 1860.36 | 1854.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 13:00:00 | 1844.50 | 1860.36 | 1854.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 1842.45 | 1856.78 | 1853.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:15:00 | 1838.30 | 1856.78 | 1853.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1848.35 | 1855.23 | 1853.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:00:00 | 1848.35 | 1855.23 | 1853.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1861.05 | 1856.40 | 1854.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 12:30:00 | 1865.95 | 1859.18 | 1855.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 1823.65 | 1852.68 | 1854.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 1823.65 | 1852.68 | 1854.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 11:15:00 | 1818.00 | 1841.83 | 1848.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 10:15:00 | 1834.50 | 1832.50 | 1839.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-11 11:00:00 | 1834.50 | 1832.50 | 1839.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 1847.50 | 1835.50 | 1840.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:45:00 | 1852.80 | 1835.50 | 1840.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1834.15 | 1835.23 | 1840.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:00:00 | 1833.10 | 1834.80 | 1839.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:30:00 | 1827.55 | 1831.77 | 1836.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 1853.40 | 1835.35 | 1836.09 | SL hit (close>static) qty=1.00 sl=1847.50 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 1849.00 | 1838.08 | 1837.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1880.00 | 1846.46 | 1841.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 1860.30 | 1879.20 | 1864.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 1860.30 | 1879.20 | 1864.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1860.30 | 1879.20 | 1864.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 1860.30 | 1879.20 | 1864.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1844.45 | 1872.25 | 1863.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 1844.45 | 1872.25 | 1863.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1851.10 | 1868.02 | 1861.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:30:00 | 1854.40 | 1868.02 | 1861.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1867.00 | 1862.04 | 1860.61 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 1855.50 | 1859.36 | 1859.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 13:15:00 | 1848.55 | 1857.19 | 1858.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 1866.00 | 1856.38 | 1857.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 1866.00 | 1856.38 | 1857.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1866.00 | 1856.38 | 1857.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 1863.05 | 1856.38 | 1857.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 10:15:00 | 1874.00 | 1859.91 | 1859.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 13:15:00 | 1885.00 | 1869.22 | 1863.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 15:15:00 | 1917.05 | 1919.60 | 1908.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 09:15:00 | 1911.80 | 1919.60 | 1908.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 13:15:00 | 1907.00 | 1915.75 | 1910.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:00:00 | 1907.00 | 1915.75 | 1910.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1905.65 | 1913.73 | 1910.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 1928.50 | 1913.83 | 1911.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 09:15:00 | 1945.50 | 1973.93 | 1976.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 1945.50 | 1973.93 | 1976.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 1939.35 | 1967.01 | 1973.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 11:15:00 | 1934.90 | 1925.00 | 1942.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 12:00:00 | 1934.90 | 1925.00 | 1942.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1877.30 | 1861.90 | 1877.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 1877.30 | 1861.90 | 1877.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1862.00 | 1861.92 | 1875.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 09:30:00 | 1857.90 | 1872.32 | 1874.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-11 13:15:00 | 1879.00 | 1875.36 | 1875.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 1879.00 | 1875.36 | 1875.10 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 09:15:00 | 1863.30 | 1873.63 | 1874.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 1854.45 | 1867.78 | 1870.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 09:15:00 | 1864.65 | 1862.93 | 1866.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 09:15:00 | 1864.65 | 1862.93 | 1866.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1864.65 | 1862.93 | 1866.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 1869.70 | 1862.93 | 1866.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 1862.00 | 1862.74 | 1866.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 1840.35 | 1856.36 | 1861.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 1748.33 | 1772.34 | 1794.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 09:15:00 | 1761.30 | 1749.15 | 1771.02 | SL hit (close>ema200) qty=0.50 sl=1749.15 alert=retest2 |

### Cycle 104 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 1765.60 | 1735.92 | 1732.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 13:15:00 | 1768.40 | 1742.42 | 1736.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 1753.80 | 1754.27 | 1747.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 14:15:00 | 1753.80 | 1754.27 | 1747.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 1753.80 | 1754.27 | 1747.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 1753.80 | 1754.27 | 1747.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1745.10 | 1752.08 | 1747.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 1739.20 | 1752.08 | 1747.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1726.45 | 1746.95 | 1745.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 1726.45 | 1746.95 | 1745.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 13:15:00 | 1739.85 | 1744.30 | 1744.59 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1751.00 | 1745.64 | 1745.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1767.55 | 1750.40 | 1747.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1723.20 | 1746.50 | 1746.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1723.20 | 1746.50 | 1746.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1723.20 | 1746.50 | 1746.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1723.20 | 1746.50 | 1746.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1708.00 | 1738.80 | 1742.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 12:15:00 | 1702.40 | 1727.64 | 1736.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1738.50 | 1715.21 | 1723.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 1738.50 | 1715.21 | 1723.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 1738.50 | 1715.21 | 1723.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 1738.50 | 1715.21 | 1723.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1734.85 | 1719.14 | 1724.46 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 1739.05 | 1729.30 | 1728.28 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1711.55 | 1730.35 | 1731.11 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 10:15:00 | 1735.50 | 1730.18 | 1730.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 13:15:00 | 1738.50 | 1732.43 | 1731.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 11:15:00 | 1736.50 | 1737.96 | 1734.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 12:00:00 | 1736.50 | 1737.96 | 1734.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1729.20 | 1736.21 | 1734.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 13:00:00 | 1729.20 | 1736.21 | 1734.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 13:15:00 | 1716.35 | 1732.24 | 1732.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 10:15:00 | 1712.75 | 1722.77 | 1727.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1583.25 | 1578.99 | 1596.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 1583.25 | 1578.99 | 1596.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 1595.00 | 1583.76 | 1594.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:45:00 | 1592.00 | 1583.76 | 1594.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1599.30 | 1586.87 | 1594.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 1599.30 | 1586.87 | 1594.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1600.70 | 1589.64 | 1595.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:15:00 | 1603.05 | 1589.64 | 1595.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 1603.05 | 1592.32 | 1596.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 1625.70 | 1592.32 | 1596.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1631.10 | 1604.50 | 1601.25 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 1594.70 | 1602.29 | 1602.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1586.05 | 1595.92 | 1598.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 1580.90 | 1580.85 | 1585.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:15:00 | 1585.90 | 1580.85 | 1585.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 1584.10 | 1581.50 | 1584.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 1587.65 | 1581.50 | 1584.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1584.40 | 1582.08 | 1584.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:15:00 | 1590.20 | 1582.08 | 1584.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1601.05 | 1585.87 | 1586.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:00:00 | 1601.05 | 1585.87 | 1586.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2024-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 14:15:00 | 1595.85 | 1587.87 | 1587.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 11:15:00 | 1604.35 | 1596.41 | 1591.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1618.50 | 1620.37 | 1610.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:45:00 | 1623.25 | 1620.37 | 1610.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1611.15 | 1618.53 | 1610.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 1610.70 | 1618.53 | 1610.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 1629.90 | 1620.80 | 1612.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:15:00 | 1630.40 | 1620.80 | 1612.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1648.50 | 1632.77 | 1620.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 11:30:00 | 1633.40 | 1636.16 | 1626.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 1634.90 | 1632.56 | 1629.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 1641.45 | 1634.34 | 1630.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:30:00 | 1644.35 | 1635.97 | 1631.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:15:00 | 1643.95 | 1635.97 | 1631.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1648.30 | 1636.76 | 1632.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:00:00 | 1643.70 | 1669.25 | 1667.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1680.25 | 1670.13 | 1668.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-17 09:15:00 | 1641.60 | 1666.22 | 1668.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 1641.60 | 1666.22 | 1668.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 09:15:00 | 1626.05 | 1645.84 | 1655.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 1600.00 | 1598.23 | 1614.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-20 10:45:00 | 1600.00 | 1598.23 | 1614.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1572.20 | 1565.09 | 1570.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:00:00 | 1572.20 | 1565.09 | 1570.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1580.45 | 1568.16 | 1571.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 10:30:00 | 1580.15 | 1568.16 | 1571.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1577.00 | 1569.93 | 1571.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 1581.30 | 1569.93 | 1571.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 1585.35 | 1573.01 | 1572.90 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 1569.60 | 1573.09 | 1573.27 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 11:15:00 | 1583.85 | 1575.24 | 1574.23 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 1563.60 | 1572.72 | 1573.25 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1579.35 | 1574.05 | 1573.80 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 1562.50 | 1571.56 | 1572.70 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 1581.20 | 1572.09 | 1571.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 1620.10 | 1584.56 | 1578.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 1695.15 | 1698.80 | 1671.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 1695.15 | 1698.80 | 1671.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1680.75 | 1690.40 | 1675.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:30:00 | 1677.25 | 1690.40 | 1675.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1684.00 | 1688.89 | 1677.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 1684.00 | 1688.89 | 1677.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1682.95 | 1690.14 | 1682.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 1682.95 | 1690.14 | 1682.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1680.05 | 1688.12 | 1682.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 1672.45 | 1688.12 | 1682.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1693.90 | 1689.28 | 1683.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 1687.80 | 1689.28 | 1683.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1683.45 | 1688.11 | 1683.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 1683.45 | 1688.11 | 1683.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 11:15:00 | 1680.90 | 1686.67 | 1683.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:00:00 | 1680.90 | 1686.67 | 1683.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 1685.40 | 1686.41 | 1683.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 12:30:00 | 1683.95 | 1686.41 | 1683.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1703.60 | 1689.85 | 1685.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:00:00 | 1712.50 | 1694.77 | 1690.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:45:00 | 1707.30 | 1696.58 | 1691.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 1708.25 | 1697.94 | 1692.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 1708.40 | 1698.68 | 1693.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 1692.00 | 1697.92 | 1694.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 1683.60 | 1697.92 | 1694.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 1682.75 | 1694.89 | 1693.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 10:15:00 | 1682.75 | 1694.89 | 1693.40 | SL hit (close<static) qty=1.00 sl=1683.00 alert=retest2 |

### Cycle 123 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 1682.30 | 1692.37 | 1692.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 1680.00 | 1688.56 | 1690.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1694.25 | 1685.84 | 1688.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1694.25 | 1685.84 | 1688.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1694.25 | 1685.84 | 1688.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1708.20 | 1685.84 | 1688.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2025-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 10:15:00 | 1710.60 | 1690.79 | 1690.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 15:15:00 | 1727.00 | 1705.64 | 1698.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 1666.10 | 1697.73 | 1695.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 09:15:00 | 1666.10 | 1697.73 | 1695.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1666.10 | 1697.73 | 1695.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 1666.10 | 1697.73 | 1695.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 10:15:00 | 1675.60 | 1693.31 | 1693.77 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-01-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 13:15:00 | 1701.80 | 1689.44 | 1688.46 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1687.10 | 1688.71 | 1688.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 1681.60 | 1687.29 | 1688.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 10:15:00 | 1695.40 | 1686.69 | 1687.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 10:15:00 | 1695.40 | 1686.69 | 1687.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1695.40 | 1686.69 | 1687.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 1700.55 | 1686.69 | 1687.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 11:15:00 | 1719.35 | 1693.23 | 1690.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 12:15:00 | 1734.90 | 1701.56 | 1694.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 14:15:00 | 1716.65 | 1728.01 | 1716.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 14:15:00 | 1716.65 | 1728.01 | 1716.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1716.65 | 1728.01 | 1716.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:45:00 | 1716.30 | 1728.01 | 1716.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1712.00 | 1724.81 | 1716.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1713.00 | 1724.81 | 1716.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 1721.95 | 1724.24 | 1716.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 13:45:00 | 1728.50 | 1724.66 | 1719.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 1728.65 | 1734.83 | 1735.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1728.65 | 1734.83 | 1735.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 1704.95 | 1728.85 | 1732.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 1726.25 | 1718.75 | 1724.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 09:15:00 | 1726.25 | 1718.75 | 1724.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1726.25 | 1718.75 | 1724.95 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 1759.60 | 1732.83 | 1730.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 13:15:00 | 1763.75 | 1739.01 | 1733.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 12:15:00 | 1773.00 | 1793.75 | 1777.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 12:15:00 | 1773.00 | 1793.75 | 1777.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 1773.00 | 1793.75 | 1777.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 1773.00 | 1793.75 | 1777.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1745.25 | 1784.05 | 1774.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 1745.25 | 1784.05 | 1774.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1743.85 | 1776.01 | 1772.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 1743.85 | 1776.01 | 1772.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-31 09:15:00 | 1730.00 | 1762.64 | 1766.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 12:15:00 | 1693.40 | 1729.85 | 1743.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 13:15:00 | 1751.55 | 1734.19 | 1744.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 13:15:00 | 1751.55 | 1734.19 | 1744.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1751.55 | 1734.19 | 1744.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:45:00 | 1752.30 | 1734.19 | 1744.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 14:15:00 | 1750.00 | 1737.35 | 1744.74 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 09:15:00 | 1776.90 | 1750.33 | 1749.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 10:15:00 | 1796.40 | 1759.54 | 1753.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 10:15:00 | 1779.85 | 1783.08 | 1771.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 11:00:00 | 1779.85 | 1783.08 | 1771.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1781.95 | 1782.19 | 1772.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 13:15:00 | 1785.00 | 1782.19 | 1772.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 12:30:00 | 1786.50 | 1789.95 | 1787.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 12:15:00 | 1781.60 | 1791.27 | 1792.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 12:15:00 | 1781.60 | 1791.27 | 1792.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 1770.00 | 1785.29 | 1788.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1760.85 | 1758.38 | 1770.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 11:00:00 | 1760.85 | 1758.38 | 1770.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1774.00 | 1761.50 | 1771.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1774.00 | 1761.50 | 1771.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1792.05 | 1767.61 | 1773.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1792.05 | 1767.61 | 1773.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1783.85 | 1770.86 | 1774.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 1793.35 | 1770.86 | 1774.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — BUY (started 2025-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 15:15:00 | 1792.25 | 1778.14 | 1776.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 1850.50 | 1792.61 | 1783.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 11:15:00 | 1827.30 | 1834.45 | 1816.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-14 12:00:00 | 1827.30 | 1834.45 | 1816.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 1879.50 | 1887.91 | 1877.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:00:00 | 1879.50 | 1887.91 | 1877.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 13:15:00 | 1880.10 | 1886.34 | 1877.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 13:30:00 | 1878.05 | 1886.34 | 1877.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 1876.60 | 1884.40 | 1877.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 14:30:00 | 1872.65 | 1884.40 | 1877.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 1872.10 | 1881.94 | 1877.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 1866.05 | 1881.94 | 1877.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 10:15:00 | 1874.35 | 1878.83 | 1876.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 10:45:00 | 1870.40 | 1878.83 | 1876.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 1862.60 | 1875.58 | 1875.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 12:00:00 | 1862.60 | 1875.58 | 1875.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 1868.50 | 1874.17 | 1874.67 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 1882.95 | 1876.17 | 1875.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-21 10:15:00 | 1884.70 | 1879.08 | 1877.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 11:15:00 | 1878.15 | 1878.89 | 1877.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 11:15:00 | 1878.15 | 1878.89 | 1877.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 1878.15 | 1878.89 | 1877.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 12:00:00 | 1878.15 | 1878.89 | 1877.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 1875.00 | 1878.11 | 1877.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 1875.00 | 1878.11 | 1877.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1876.05 | 1877.70 | 1876.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 1873.40 | 1877.70 | 1876.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1873.25 | 1877.41 | 1876.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 1885.40 | 1877.41 | 1876.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1874.50 | 1876.83 | 1876.74 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1869.05 | 1875.27 | 1876.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 1863.95 | 1873.01 | 1874.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1890.05 | 1870.60 | 1872.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 1890.05 | 1870.60 | 1872.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1890.05 | 1870.60 | 1872.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 1887.05 | 1870.60 | 1872.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1879.55 | 1872.39 | 1872.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:30:00 | 1887.65 | 1872.39 | 1872.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 1882.00 | 1874.31 | 1873.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 1922.05 | 1883.94 | 1878.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 1894.65 | 1908.97 | 1897.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 1894.65 | 1908.97 | 1897.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 1894.65 | 1908.97 | 1897.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 1894.65 | 1908.97 | 1897.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 10:15:00 | 1886.25 | 1904.43 | 1896.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:45:00 | 1883.90 | 1904.43 | 1896.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 11:15:00 | 1894.65 | 1902.47 | 1896.10 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-02-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 13:15:00 | 1870.95 | 1890.85 | 1891.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 1837.60 | 1874.29 | 1883.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 13:15:00 | 1800.50 | 1796.26 | 1815.17 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-06 09:15:00 | 1793.00 | 1799.16 | 1813.26 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 1814.55 | 1802.23 | 1813.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-06 09:15:00 | 1814.55 | 1802.23 | 1813.38 | SL hit (close>ema400) qty=1.00 sl=1813.38 alert=retest1 |

### Cycle 140 — BUY (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 13:15:00 | 1850.00 | 1821.86 | 1819.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 1861.90 | 1835.80 | 1827.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 1839.00 | 1843.36 | 1835.73 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1876.25 | 1843.36 | 1835.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 1848.25 | 1855.62 | 1846.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 1845.55 | 1855.62 | 1846.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1840.90 | 1852.68 | 1845.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 1840.90 | 1852.68 | 1845.68 | SL hit (close<ema400) qty=1.00 sl=1845.68 alert=retest1 |

### Cycle 141 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 1805.55 | 1834.68 | 1838.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 1795.05 | 1821.12 | 1831.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1822.65 | 1815.13 | 1823.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 11:00:00 | 1822.65 | 1815.13 | 1823.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1808.85 | 1813.87 | 1822.43 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 1864.05 | 1822.43 | 1821.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 13:15:00 | 1878.95 | 1849.88 | 1835.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 09:15:00 | 1846.70 | 1856.03 | 1842.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 1846.70 | 1856.03 | 1842.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1846.70 | 1856.03 | 1842.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:45:00 | 1843.75 | 1856.03 | 1842.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1843.75 | 1853.58 | 1842.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 1838.00 | 1853.58 | 1842.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 1839.95 | 1850.85 | 1842.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:30:00 | 1839.65 | 1850.85 | 1842.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 1847.65 | 1850.21 | 1843.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:30:00 | 1837.45 | 1850.21 | 1843.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 13:15:00 | 1840.95 | 1848.36 | 1842.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:30:00 | 1844.00 | 1848.36 | 1842.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 14:15:00 | 1844.85 | 1847.66 | 1843.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 09:15:00 | 1859.20 | 1847.43 | 1843.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-19 11:30:00 | 1846.60 | 1848.78 | 1845.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-20 09:15:00 | 1828.15 | 1842.28 | 1843.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-20 09:15:00 | 1828.15 | 1842.28 | 1843.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-20 10:15:00 | 1824.90 | 1838.80 | 1841.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 12:15:00 | 1847.40 | 1839.60 | 1841.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 12:15:00 | 1847.40 | 1839.60 | 1841.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 12:15:00 | 1847.40 | 1839.60 | 1841.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 13:00:00 | 1847.40 | 1839.60 | 1841.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 1847.55 | 1841.19 | 1841.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 14:00:00 | 1847.55 | 1841.19 | 1841.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 14:15:00 | 1854.50 | 1843.85 | 1843.12 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 11:15:00 | 1838.45 | 1842.18 | 1842.59 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 1849.60 | 1842.76 | 1842.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 1874.35 | 1848.95 | 1845.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 15:15:00 | 1935.00 | 1941.23 | 1923.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-27 09:15:00 | 1950.60 | 1941.23 | 1923.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1987.80 | 1997.54 | 1979.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 1994.80 | 1997.54 | 1979.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1942.65 | 1986.56 | 1976.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1942.65 | 1986.56 | 1976.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1939.35 | 1977.12 | 1973.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:30:00 | 1936.90 | 1977.12 | 1973.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 1947.40 | 1967.79 | 1969.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 1934.35 | 1961.10 | 1966.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 1940.60 | 1940.14 | 1950.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-03 10:15:00 | 1947.35 | 1940.14 | 1950.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1919.50 | 1936.02 | 1947.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 1912.90 | 1930.29 | 1943.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 1908.85 | 1922.93 | 1934.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 11:30:00 | 1912.60 | 1921.12 | 1931.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 14:00:00 | 1915.25 | 1919.58 | 1929.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1901.60 | 1870.06 | 1889.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:45:00 | 1906.35 | 1870.06 | 1889.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 1889.90 | 1874.03 | 1889.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-09 15:15:00 | 1900.00 | 1893.88 | 1893.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-04-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 15:15:00 | 1900.00 | 1893.88 | 1893.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1949.35 | 1904.97 | 1898.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 1955.10 | 1963.20 | 1948.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 1955.10 | 1963.20 | 1948.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1970.20 | 1963.51 | 1952.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 1956.00 | 1963.51 | 1952.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1974.00 | 1967.05 | 1956.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 1986.80 | 1971.14 | 1960.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 2033.80 | 2078.75 | 2079.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 2033.80 | 2078.75 | 2079.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 1937.90 | 2036.62 | 2050.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 12:15:00 | 1954.40 | 1953.59 | 1985.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 1954.40 | 1953.59 | 1985.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 1964.50 | 1955.77 | 1983.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:30:00 | 1968.00 | 1955.77 | 1983.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 2010.50 | 1966.87 | 1981.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 2009.50 | 1966.87 | 1981.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 2010.20 | 1975.54 | 1984.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:30:00 | 2005.50 | 1982.39 | 1986.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:15:00 | 2005.90 | 1982.39 | 1986.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 2029.00 | 1995.99 | 1992.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 2029.00 | 1995.99 | 1992.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 13:15:00 | 2043.50 | 2022.19 | 2009.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2014.50 | 2024.17 | 2013.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 2014.50 | 2024.17 | 2013.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 2014.50 | 2024.17 | 2013.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:15:00 | 2015.70 | 2024.17 | 2013.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 2020.10 | 2023.35 | 2014.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:45:00 | 2012.60 | 2023.35 | 2014.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 2025.80 | 2022.90 | 2016.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:30:00 | 2016.50 | 2022.90 | 2016.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 2024.60 | 2028.38 | 2022.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:45:00 | 2022.70 | 2028.38 | 2022.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 2020.40 | 2026.78 | 2021.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:00:00 | 2020.40 | 2026.78 | 2021.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 2009.10 | 2023.25 | 2020.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:30:00 | 1998.90 | 2023.25 | 2020.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 2009.40 | 2020.48 | 2019.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 2009.40 | 2020.48 | 2019.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 2000.00 | 2016.38 | 2017.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 1981.90 | 2009.49 | 2014.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 2033.70 | 1991.88 | 1999.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 2033.70 | 1991.88 | 1999.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2033.70 | 1991.88 | 1999.20 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2036.00 | 2007.48 | 2005.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2045.70 | 2020.32 | 2011.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 2024.00 | 2031.08 | 2022.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 2024.00 | 2031.08 | 2022.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 2024.00 | 2031.08 | 2022.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 2024.00 | 2031.08 | 2022.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 2020.90 | 2029.04 | 2022.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:15:00 | 2016.90 | 2029.04 | 2022.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 2017.00 | 2026.64 | 2022.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 2022.90 | 2026.64 | 2022.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 11:15:00 | 2023.20 | 2026.63 | 2023.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 2007.40 | 2019.17 | 2020.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 09:15:00 | 2007.40 | 2019.17 | 2020.63 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 2046.70 | 2026.01 | 2023.24 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 2020.50 | 2031.43 | 2031.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 2011.30 | 2027.41 | 2029.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2020.30 | 2017.85 | 2023.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 2020.30 | 2017.85 | 2023.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2033.00 | 2020.88 | 2024.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 2034.50 | 2020.88 | 2024.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 2019.80 | 2020.66 | 2024.06 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 2041.00 | 2026.31 | 2025.91 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 2013.80 | 2025.09 | 2025.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 2006.00 | 2021.27 | 2023.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 2018.30 | 2010.40 | 2015.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 2018.30 | 2010.40 | 2015.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2018.30 | 2010.40 | 2015.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:45:00 | 2015.70 | 2010.40 | 2015.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2022.50 | 2012.82 | 2016.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 2024.50 | 2012.82 | 2016.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 2019.90 | 2014.24 | 2016.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 2022.90 | 2014.24 | 2016.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 2040.70 | 2019.53 | 2018.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2069.00 | 2037.16 | 2028.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 2030.00 | 2045.79 | 2038.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 2030.00 | 2045.79 | 2038.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2030.00 | 2045.79 | 2038.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 2052.00 | 2044.55 | 2038.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 14:15:00 | 2028.00 | 2036.50 | 2036.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 2028.00 | 2036.50 | 2036.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 2011.20 | 2025.10 | 2030.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 2020.00 | 2014.89 | 2021.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:15:00 | 2022.00 | 2014.89 | 2021.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 2022.20 | 2016.35 | 2021.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 2025.60 | 2016.35 | 2021.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 2020.20 | 2017.12 | 2021.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:15:00 | 2024.90 | 2017.12 | 2021.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 2019.00 | 2017.50 | 2021.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:30:00 | 2018.70 | 2017.50 | 2021.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 2017.50 | 2016.41 | 2019.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:30:00 | 2015.60 | 2016.41 | 2019.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 2011.50 | 2015.43 | 2018.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1999.10 | 2015.43 | 2018.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 14:15:00 | 2031.70 | 2019.29 | 2018.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 2031.70 | 2019.29 | 2018.81 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2009.90 | 2018.51 | 2018.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 1994.00 | 2013.61 | 2016.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 2000.00 | 1998.34 | 2005.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 2000.00 | 1998.34 | 2005.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2000.00 | 1998.34 | 2005.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 12:15:00 | 1991.00 | 1997.31 | 2003.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 13:00:00 | 1957.00 | 1989.25 | 1999.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 11:15:00 | 2011.10 | 1964.57 | 1969.12 | SL hit (close>static) qty=1.00 sl=2007.80 alert=retest2 |

### Cycle 162 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 2003.50 | 1972.36 | 1972.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2036.90 | 1990.71 | 1981.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 2000.60 | 2010.32 | 1999.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 2000.60 | 2010.32 | 1999.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2000.60 | 2010.32 | 1999.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1995.40 | 2010.32 | 1999.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1998.20 | 2007.89 | 1998.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:30:00 | 1998.70 | 2007.89 | 1998.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 2002.70 | 2006.85 | 1999.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 2011.50 | 2006.78 | 1999.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:00:00 | 2013.50 | 2005.15 | 2001.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:45:00 | 2009.00 | 2008.73 | 2003.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 2008.40 | 2007.82 | 2003.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2013.30 | 2008.92 | 2004.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 2043.00 | 2009.54 | 2005.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1997.00 | 2022.18 | 2017.94 | SL hit (close<static) qty=1.00 sl=2002.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1995.90 | 2011.90 | 2013.69 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 2026.90 | 2015.49 | 2013.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 2036.00 | 2020.31 | 2016.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 2011.00 | 2020.79 | 2017.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 2011.00 | 2020.79 | 2017.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2011.00 | 2020.79 | 2017.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:30:00 | 2012.00 | 2020.79 | 2017.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 2021.00 | 2020.83 | 2018.16 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 2001.30 | 2013.75 | 2015.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 1995.00 | 2006.67 | 2010.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1984.40 | 1977.35 | 1986.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1984.40 | 1977.35 | 1986.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1984.40 | 1977.35 | 1986.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1984.40 | 1977.35 | 1986.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1980.40 | 1977.96 | 1985.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1969.70 | 1977.96 | 1985.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 1976.20 | 1977.95 | 1984.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1976.80 | 1979.92 | 1984.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 1993.00 | 1978.56 | 1981.50 | SL hit (close>static) qty=1.00 sl=1989.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1998.40 | 1986.18 | 1984.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2012.50 | 1992.86 | 1988.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 2003.90 | 2005.70 | 1996.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 2006.80 | 2005.70 | 1996.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 2017.60 | 2032.09 | 2021.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 2008.60 | 2032.09 | 2021.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 2033.50 | 2032.37 | 2022.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:30:00 | 2048.40 | 2034.70 | 2026.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 2050.30 | 2034.96 | 2027.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 10:00:00 | 2049.00 | 2037.76 | 2029.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:45:00 | 2048.00 | 2041.50 | 2032.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 2040.90 | 2041.33 | 2034.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 2035.00 | 2041.33 | 2034.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 2026.60 | 2047.15 | 2043.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 2024.80 | 2047.15 | 2043.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 2019.60 | 2041.64 | 2041.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 2019.60 | 2041.64 | 2041.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 2011.00 | 2035.51 | 2038.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 2010.40 | 1994.55 | 2007.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 2010.40 | 1994.55 | 2007.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 2010.40 | 1994.55 | 2007.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:45:00 | 2018.80 | 1994.55 | 2007.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 2001.50 | 1995.94 | 2006.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 1993.80 | 1995.35 | 2005.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1997.20 | 1997.86 | 2001.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 2007.10 | 2003.21 | 2003.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 09:15:00 | 2007.10 | 2003.21 | 2003.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 2019.90 | 2007.87 | 2005.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 2023.10 | 2024.01 | 2016.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 2034.40 | 2024.01 | 2016.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 2020.90 | 2031.70 | 2025.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 2020.90 | 2031.70 | 2025.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 2006.40 | 2026.64 | 2024.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 2006.40 | 2026.64 | 2024.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 2005.10 | 2022.33 | 2022.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 10:15:00 | 1999.90 | 2011.95 | 2016.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 2013.70 | 2005.26 | 2010.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 2013.70 | 2005.26 | 2010.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 2013.70 | 2005.26 | 2010.09 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2020.70 | 2013.05 | 2012.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2035.90 | 2017.62 | 2014.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 2025.00 | 2025.04 | 2019.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 2022.30 | 2024.49 | 2019.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 2022.30 | 2024.49 | 2019.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 2022.30 | 2024.49 | 2019.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 2021.00 | 2023.79 | 2019.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:00:00 | 2032.00 | 2025.43 | 2021.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:45:00 | 2030.30 | 2029.89 | 2025.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:30:00 | 2032.00 | 2028.70 | 2025.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 2030.00 | 2028.70 | 2025.32 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 2031.40 | 2029.24 | 2025.87 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 2015.60 | 2025.54 | 2025.36 | SL hit (close<static) qty=1.00 sl=2019.60 alert=retest2 |

### Cycle 171 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 2008.50 | 2022.14 | 2023.82 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 2040.00 | 2027.19 | 2025.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 2049.40 | 2031.63 | 2027.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 2028.40 | 2040.83 | 2035.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 2028.40 | 2040.83 | 2035.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 2028.40 | 2040.83 | 2035.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:00:00 | 2055.00 | 2041.99 | 2037.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 15:15:00 | 2030.00 | 2041.81 | 2042.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 2030.00 | 2041.81 | 2042.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1943.50 | 2022.15 | 2033.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2012.80 | 1992.61 | 2007.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2012.80 | 1992.61 | 2007.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2012.80 | 1992.61 | 2007.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 2010.10 | 1992.61 | 2007.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1994.20 | 1992.93 | 2006.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 1990.60 | 1992.42 | 2003.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 13:15:00 | 1891.07 | 1911.65 | 1920.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1912.80 | 1911.88 | 1919.93 | SL hit (close>ema200) qty=0.50 sl=1911.88 alert=retest2 |

### Cycle 174 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1921.10 | 1918.12 | 1917.86 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1911.80 | 1916.86 | 1917.31 | EMA200 below EMA400 |

### Cycle 176 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1925.90 | 1918.99 | 1918.22 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1908.30 | 1916.55 | 1917.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 15:15:00 | 1905.00 | 1914.24 | 1916.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 12:15:00 | 1912.50 | 1911.15 | 1913.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 12:15:00 | 1912.50 | 1911.15 | 1913.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1912.50 | 1911.15 | 1913.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 1912.50 | 1911.15 | 1913.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1919.00 | 1912.72 | 1914.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1919.00 | 1912.72 | 1914.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1914.10 | 1913.00 | 1914.18 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 1933.50 | 1917.74 | 1916.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 2010.90 | 1941.10 | 1928.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1978.50 | 1982.73 | 1961.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 09:45:00 | 1980.90 | 1982.73 | 1961.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1971.80 | 1975.27 | 1965.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:30:00 | 1968.20 | 1975.27 | 1965.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1958.90 | 1971.78 | 1965.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 09:30:00 | 1960.50 | 1971.78 | 1965.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 1958.50 | 1969.12 | 1964.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 1958.50 | 1969.12 | 1964.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1956.40 | 1961.34 | 1961.96 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1986.20 | 1965.94 | 1963.92 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 1967.50 | 1969.62 | 1969.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 1955.00 | 1966.70 | 1968.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 13:15:00 | 1965.90 | 1963.91 | 1966.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 13:15:00 | 1965.90 | 1963.91 | 1966.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1965.90 | 1963.91 | 1966.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:00:00 | 1965.90 | 1963.91 | 1966.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1963.20 | 1963.77 | 1965.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1948.40 | 1963.80 | 1965.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 1939.30 | 1932.54 | 1931.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 1939.30 | 1932.54 | 1931.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 1945.10 | 1935.05 | 1933.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 2007.70 | 2014.26 | 2004.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 15:00:00 | 2007.70 | 2014.26 | 2004.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2033.00 | 2035.92 | 2029.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 2033.00 | 2035.92 | 2029.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 2036.30 | 2036.00 | 2029.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 2030.00 | 2036.00 | 2029.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 2086.20 | 2083.62 | 2076.20 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 2057.30 | 2073.20 | 2073.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 09:15:00 | 2049.40 | 2064.21 | 2068.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 11:15:00 | 2066.40 | 2064.28 | 2067.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 2066.40 | 2064.28 | 2067.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2063.30 | 2063.91 | 2067.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:45:00 | 2065.30 | 2063.91 | 2067.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 2067.50 | 2064.63 | 2067.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 15:00:00 | 2067.50 | 2064.63 | 2067.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 2064.50 | 2064.60 | 2066.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 2071.00 | 2064.60 | 2066.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 2069.00 | 2065.48 | 2067.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:30:00 | 2061.10 | 2063.83 | 2066.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 2061.00 | 2062.42 | 2064.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 2073.10 | 2066.53 | 2066.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 2073.10 | 2066.53 | 2066.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 09:15:00 | 2085.00 | 2073.55 | 2070.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 2081.00 | 2081.06 | 2076.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 2081.00 | 2081.06 | 2076.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 2081.00 | 2081.06 | 2076.70 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 2068.10 | 2074.64 | 2074.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 2064.20 | 2071.82 | 2073.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 2009.80 | 2008.06 | 2022.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 14:00:00 | 2009.80 | 2008.06 | 2022.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 2025.10 | 2011.46 | 2022.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 2025.10 | 2011.46 | 2022.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 2012.00 | 2011.57 | 2021.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 2026.70 | 2011.57 | 2021.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 2015.30 | 2012.32 | 2021.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:45:00 | 2009.80 | 2012.86 | 2020.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 2021.80 | 2005.53 | 2004.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 2021.80 | 2005.53 | 2004.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 2030.00 | 2010.42 | 2006.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 2031.10 | 2037.70 | 2027.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 15:00:00 | 2031.10 | 2037.70 | 2027.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 2025.60 | 2034.05 | 2027.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 2025.60 | 2034.05 | 2027.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 2012.80 | 2029.80 | 2025.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 2012.80 | 2029.80 | 2025.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 2009.90 | 2025.82 | 2024.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 2009.90 | 2025.82 | 2024.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 2014.30 | 2022.02 | 2022.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 2010.10 | 2019.64 | 2021.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 2020.20 | 2016.85 | 2019.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 2020.20 | 2016.85 | 2019.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2020.20 | 2016.85 | 2019.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2020.20 | 2016.85 | 2019.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2022.00 | 2017.88 | 2019.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:30:00 | 2024.50 | 2017.88 | 2019.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 2018.00 | 2017.91 | 2019.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 2015.40 | 2017.91 | 2019.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 2015.90 | 2017.51 | 2019.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:00:00 | 2015.20 | 2017.58 | 2018.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 15:15:00 | 2023.90 | 2015.30 | 2014.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 2023.90 | 2015.30 | 2014.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 2025.30 | 2017.30 | 2015.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 2014.60 | 2017.90 | 2016.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 11:15:00 | 2014.60 | 2017.90 | 2016.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 2014.60 | 2017.90 | 2016.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 2014.60 | 2017.90 | 2016.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 2015.30 | 2017.38 | 2016.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:45:00 | 2009.70 | 2017.38 | 2016.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 2019.00 | 2017.15 | 2016.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 2038.40 | 2017.70 | 2016.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 2131.90 | 2153.02 | 2155.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 2131.90 | 2153.02 | 2155.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 2129.90 | 2141.64 | 2148.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 2140.00 | 2135.33 | 2141.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 15:15:00 | 2140.00 | 2135.33 | 2141.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 2140.00 | 2135.33 | 2141.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 2132.10 | 2135.33 | 2141.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2129.00 | 2134.06 | 2140.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:45:00 | 2121.10 | 2132.09 | 2138.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 11:45:00 | 2116.00 | 2127.85 | 2136.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:00:00 | 2119.20 | 2119.17 | 2127.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 14:15:00 | 2100.30 | 2077.28 | 2077.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 2100.30 | 2077.28 | 2077.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 15:15:00 | 2110.00 | 2083.82 | 2080.23 | Break + close above crossover candle high |

### Cycle 191 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1978.70 | 2082.75 | 2086.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 1963.80 | 2058.96 | 2075.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 2010.30 | 2008.09 | 2034.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:30:00 | 2013.10 | 2008.09 | 2034.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 12:15:00 | 2028.50 | 2012.43 | 2031.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:00:00 | 2028.50 | 2012.43 | 2031.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 2034.90 | 2016.92 | 2031.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 13:30:00 | 2033.00 | 2016.92 | 2031.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 2035.00 | 2020.54 | 2032.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:30:00 | 2033.20 | 2020.54 | 2032.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 15:15:00 | 2039.00 | 2024.23 | 2032.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:15:00 | 2066.30 | 2024.23 | 2032.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 2060.50 | 2037.02 | 2037.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 2060.50 | 2037.02 | 2037.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 2061.70 | 2041.95 | 2039.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 14:15:00 | 2068.70 | 2055.89 | 2050.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 10:15:00 | 2056.80 | 2058.80 | 2053.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:30:00 | 2057.30 | 2058.80 | 2053.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2059.30 | 2067.09 | 2060.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 2059.30 | 2067.09 | 2060.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 2058.50 | 2065.37 | 2060.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 2058.30 | 2065.37 | 2060.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 2060.00 | 2064.30 | 2060.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 2059.90 | 2064.30 | 2060.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2055.90 | 2062.62 | 2059.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 2056.30 | 2062.62 | 2059.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 2058.80 | 2061.86 | 2059.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:30:00 | 2050.00 | 2061.86 | 2059.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 2053.30 | 2058.25 | 2058.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 2042.90 | 2055.18 | 2056.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 2050.00 | 2048.40 | 2052.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:30:00 | 2051.10 | 2048.40 | 2052.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 2049.20 | 2048.56 | 2051.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 2051.50 | 2048.56 | 2051.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 2057.30 | 2050.31 | 2052.40 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 2072.50 | 2054.75 | 2054.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 2079.30 | 2059.66 | 2056.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 2078.30 | 2081.73 | 2070.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:45:00 | 2081.50 | 2081.73 | 2070.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 2071.50 | 2079.68 | 2070.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 2071.50 | 2079.68 | 2070.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 2075.00 | 2078.75 | 2071.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 2070.10 | 2078.75 | 2071.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 2075.00 | 2078.00 | 2071.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 2073.40 | 2078.00 | 2071.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 2070.40 | 2076.48 | 2071.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 2071.00 | 2076.48 | 2071.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2053.00 | 2071.78 | 2069.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 2053.00 | 2071.78 | 2069.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 2056.90 | 2068.81 | 2068.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 2053.20 | 2068.81 | 2068.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 2058.60 | 2066.76 | 2067.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 2038.60 | 2060.45 | 2064.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 2045.10 | 2045.05 | 2052.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:45:00 | 2044.00 | 2045.05 | 2052.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 2060.00 | 2044.16 | 2049.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 2060.00 | 2044.16 | 2049.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 2061.70 | 2047.67 | 2050.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 2061.30 | 2047.67 | 2050.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 2076.00 | 2056.67 | 2054.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 2085.80 | 2065.75 | 2058.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 11:15:00 | 2092.80 | 2097.70 | 2086.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:00:00 | 2092.80 | 2097.70 | 2086.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2093.70 | 2096.90 | 2087.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 2092.60 | 2096.90 | 2087.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 2094.10 | 2095.15 | 2088.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 09:15:00 | 2097.50 | 2095.15 | 2088.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:15:00 | 2097.00 | 2094.18 | 2089.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 2081.10 | 2091.56 | 2088.65 | SL hit (close<static) qty=1.00 sl=2088.50 alert=retest2 |

### Cycle 197 — SELL (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 15:15:00 | 2082.00 | 2086.80 | 2087.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 2064.70 | 2082.38 | 2085.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 2053.90 | 2050.50 | 2060.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 09:30:00 | 2057.00 | 2050.50 | 2060.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 2060.60 | 2052.52 | 2060.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:45:00 | 2057.30 | 2052.52 | 2060.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 2050.80 | 2052.17 | 2059.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:30:00 | 2043.30 | 2050.60 | 2058.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 2048.50 | 2049.73 | 2056.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 2081.60 | 2055.91 | 2058.12 | SL hit (close>static) qty=1.00 sl=2061.10 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 2093.30 | 2063.39 | 2061.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 2098.20 | 2070.35 | 2064.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2068.30 | 2083.07 | 2074.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 2068.30 | 2083.07 | 2074.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 2068.30 | 2083.07 | 2074.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 2068.30 | 2083.07 | 2074.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 2073.90 | 2081.24 | 2074.71 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 2061.00 | 2069.49 | 2070.65 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 2089.00 | 2072.36 | 2070.49 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 2060.00 | 2070.59 | 2070.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 2057.10 | 2064.71 | 2067.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 13:15:00 | 2066.30 | 2065.03 | 2067.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 13:15:00 | 2066.30 | 2065.03 | 2067.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 2066.30 | 2065.03 | 2067.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 2066.30 | 2065.03 | 2067.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 2065.70 | 2065.16 | 2067.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:30:00 | 2066.10 | 2065.16 | 2067.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2071.00 | 2066.32 | 2067.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 2078.00 | 2066.32 | 2067.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 2074.10 | 2069.17 | 2068.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 2083.00 | 2071.93 | 2070.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2066.90 | 2075.62 | 2072.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2066.90 | 2075.62 | 2072.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2066.90 | 2075.62 | 2072.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 2064.00 | 2075.62 | 2072.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2065.60 | 2073.62 | 2072.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:30:00 | 2064.30 | 2073.62 | 2072.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2074.50 | 2074.02 | 2072.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:45:00 | 2073.30 | 2074.02 | 2072.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 13:15:00 | 2071.80 | 2073.57 | 2072.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 14:00:00 | 2071.80 | 2073.57 | 2072.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 14:15:00 | 2070.50 | 2072.96 | 2072.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 15:00:00 | 2070.50 | 2072.96 | 2072.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 2067.10 | 2071.79 | 2071.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 2046.90 | 2066.81 | 2069.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 2033.60 | 2027.33 | 2038.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 2034.40 | 2027.33 | 2038.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2034.90 | 2028.84 | 2037.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2035.00 | 2028.84 | 2037.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 2047.70 | 2031.82 | 2035.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 2052.70 | 2031.82 | 2035.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2037.00 | 2032.86 | 2035.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 2035.60 | 2032.86 | 2035.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:15:00 | 2032.60 | 2033.70 | 2036.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 2044.00 | 2038.16 | 2037.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2044.00 | 2038.16 | 2037.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2050.00 | 2040.53 | 2038.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 2040.00 | 2040.42 | 2038.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 2040.00 | 2040.42 | 2038.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 2040.00 | 2040.42 | 2038.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:45:00 | 2042.20 | 2040.42 | 2038.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 2040.30 | 2040.40 | 2038.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 2046.10 | 2040.40 | 2038.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 2058.10 | 2047.14 | 2045.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 12:15:00 | 2034.70 | 2045.26 | 2045.19 | SL hit (close<static) qty=1.00 sl=2038.80 alert=retest2 |

### Cycle 205 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 2034.60 | 2043.13 | 2044.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 2025.20 | 2037.58 | 2041.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 2006.30 | 2000.04 | 2010.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 14:00:00 | 2006.30 | 2000.04 | 2010.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2026.50 | 2005.33 | 2011.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 2026.50 | 2005.33 | 2011.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 2040.20 | 2012.30 | 2014.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 2019.70 | 2012.30 | 2014.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 11:15:00 | 2029.70 | 2016.99 | 2015.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2029.70 | 2016.99 | 2015.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 2033.20 | 2020.23 | 2017.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 12:15:00 | 2031.00 | 2031.77 | 2026.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 12:45:00 | 2031.80 | 2031.77 | 2026.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2045.90 | 2036.30 | 2030.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 2065.10 | 2038.49 | 2036.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 13:30:00 | 2050.40 | 2048.41 | 2043.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 2031.30 | 2041.44 | 2041.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 2031.30 | 2041.44 | 2041.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 2014.50 | 2030.47 | 2035.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2004.00 | 1993.91 | 2002.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 2004.00 | 1993.91 | 2002.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 2004.00 | 1993.91 | 2002.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 13:00:00 | 2004.00 | 1993.91 | 2002.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 1991.00 | 1993.33 | 2001.65 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 14:15:00 | 2010.90 | 2002.82 | 2002.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 2022.90 | 2008.51 | 2006.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 2007.30 | 2011.16 | 2008.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 12:15:00 | 2007.30 | 2011.16 | 2008.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 2007.30 | 2011.16 | 2008.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:00:00 | 2007.30 | 2011.16 | 2008.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2006.40 | 2010.20 | 2007.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 13:30:00 | 2004.10 | 2010.20 | 2007.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2002.20 | 2008.60 | 2007.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 15:00:00 | 2002.20 | 2008.60 | 2007.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2000.40 | 2005.91 | 2006.36 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 2013.40 | 2007.41 | 2007.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 12:15:00 | 2014.60 | 2009.05 | 2007.83 | Break + close above crossover candle high |

### Cycle 211 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 1979.00 | 2006.73 | 2007.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 1976.20 | 1993.51 | 2000.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 1965.00 | 1963.95 | 1975.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 1978.20 | 1963.95 | 1975.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1981.80 | 1967.52 | 1975.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1985.80 | 1967.52 | 1975.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1981.90 | 1970.40 | 1976.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 1980.30 | 1970.40 | 1976.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1977.60 | 1973.66 | 1977.02 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 1996.00 | 1982.06 | 1980.31 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 1973.70 | 1979.16 | 1979.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1952.70 | 1972.04 | 1975.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 1937.80 | 1934.21 | 1947.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 11:00:00 | 1937.80 | 1934.21 | 1947.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1938.00 | 1932.15 | 1942.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 1938.90 | 1932.15 | 1942.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1941.00 | 1933.92 | 1941.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1933.70 | 1933.92 | 1941.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 1952.90 | 1936.52 | 1940.44 | SL hit (close>static) qty=1.00 sl=1944.10 alert=retest2 |

### Cycle 214 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1950.20 | 1943.04 | 1942.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 14:15:00 | 1953.00 | 1945.03 | 1943.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1941.10 | 1945.63 | 1943.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1941.10 | 1945.63 | 1943.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1941.10 | 1945.63 | 1943.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 1940.30 | 1945.63 | 1943.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1945.00 | 1945.51 | 1943.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 1945.00 | 1945.51 | 1943.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1940.00 | 1944.40 | 1943.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1940.00 | 1944.40 | 1943.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1930.00 | 1941.52 | 1942.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1904.00 | 1926.80 | 1934.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1918.60 | 1917.00 | 1926.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 1918.60 | 1917.00 | 1926.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1930.30 | 1919.79 | 1926.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1930.30 | 1919.79 | 1926.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1934.00 | 1922.63 | 1927.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2022.00 | 1922.63 | 1927.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2010.50 | 1940.20 | 1934.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 2047.70 | 2029.40 | 2022.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 2030.00 | 2030.00 | 2023.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 12:15:00 | 2030.00 | 2030.00 | 2023.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 2030.00 | 2030.00 | 2023.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:00:00 | 2030.00 | 2030.00 | 2023.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 2025.90 | 2028.97 | 2024.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 14:45:00 | 2027.30 | 2028.97 | 2024.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 2025.30 | 2028.24 | 2024.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 2027.30 | 2028.24 | 2024.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2025.00 | 2027.59 | 2024.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:45:00 | 2033.30 | 2028.67 | 2025.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:00:00 | 2033.00 | 2029.54 | 2026.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:30:00 | 2034.80 | 2030.23 | 2026.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 2017.10 | 2025.99 | 2026.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 12:15:00 | 2017.10 | 2025.99 | 2026.72 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 09:15:00 | 2039.90 | 2027.55 | 2026.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 2046.70 | 2035.37 | 2031.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 2043.10 | 2044.48 | 2039.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 2043.10 | 2044.48 | 2039.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 2054.80 | 2046.04 | 2041.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 10:30:00 | 2062.90 | 2049.43 | 2043.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 2061.50 | 2053.58 | 2047.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 2033.60 | 2046.22 | 2046.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 2033.60 | 2046.22 | 2046.81 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 2056.20 | 2047.08 | 2046.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 2062.60 | 2050.19 | 2048.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 2045.00 | 2051.04 | 2048.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 15:15:00 | 2045.00 | 2051.04 | 2048.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 2045.00 | 2051.04 | 2048.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 2060.50 | 2051.04 | 2048.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2064.90 | 2053.81 | 2050.36 | EMA400 retest candle locked (from upside) |

### Cycle 221 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 2032.90 | 2048.41 | 2049.64 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 2071.40 | 2048.82 | 2048.47 | EMA200 above EMA400 |

### Cycle 223 — SELL (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 11:15:00 | 2047.10 | 2050.04 | 2050.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 12:15:00 | 2042.20 | 2048.47 | 2049.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1910.10 | 1896.50 | 1918.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1910.10 | 1896.50 | 1918.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 1864.40 | 1850.59 | 1863.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 1864.40 | 1850.59 | 1863.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1868.50 | 1854.17 | 1863.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 1868.50 | 1854.17 | 1863.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 1871.40 | 1857.62 | 1864.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:00:00 | 1871.40 | 1857.62 | 1864.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 1871.40 | 1862.07 | 1865.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 1854.60 | 1862.07 | 1865.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1844.30 | 1858.52 | 1863.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:30:00 | 1842.30 | 1852.99 | 1860.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 1750.18 | 1765.83 | 1790.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 1775.00 | 1754.45 | 1767.35 | SL hit (close>ema200) qty=0.50 sl=1754.45 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 1772.90 | 1770.42 | 1770.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 1780.00 | 1772.62 | 1771.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1745.20 | 1777.47 | 1776.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1745.20 | 1777.47 | 1776.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1745.20 | 1777.47 | 1776.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1745.20 | 1777.47 | 1776.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 225 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1742.20 | 1770.42 | 1773.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 1735.90 | 1763.52 | 1769.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1686.60 | 1683.86 | 1704.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1686.60 | 1683.86 | 1704.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1686.60 | 1683.86 | 1704.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1682.60 | 1681.73 | 1701.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:00:00 | 1681.10 | 1681.60 | 1699.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1750.70 | 1706.78 | 1705.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1750.70 | 1706.78 | 1705.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1763.70 | 1718.17 | 1711.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1706.40 | 1732.18 | 1723.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1706.40 | 1732.18 | 1723.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1706.40 | 1732.18 | 1723.65 | EMA400 retest candle locked (from upside) |

### Cycle 227 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 1695.10 | 1714.91 | 1717.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 1687.50 | 1709.43 | 1714.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1665.00 | 1655.89 | 1676.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1665.00 | 1655.89 | 1676.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1665.00 | 1655.89 | 1676.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:30:00 | 1661.10 | 1657.99 | 1675.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 1658.50 | 1662.86 | 1673.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1660.30 | 1662.86 | 1673.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:45:00 | 1662.10 | 1646.54 | 1648.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 1669.70 | 1651.18 | 1650.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 228 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 1669.70 | 1651.18 | 1650.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 1684.10 | 1665.09 | 1658.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1763.00 | 1769.98 | 1742.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 1763.00 | 1769.98 | 1742.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1769.60 | 1793.24 | 1775.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 1782.00 | 1786.14 | 1775.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 1796.20 | 1832.08 | 1836.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 229 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 1796.20 | 1832.08 | 1836.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1792.10 | 1812.84 | 1824.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 1776.00 | 1774.62 | 1787.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 13:45:00 | 1776.00 | 1774.62 | 1787.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 1776.60 | 1774.62 | 1784.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:00:00 | 1776.60 | 1774.62 | 1784.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1779.60 | 1774.71 | 1781.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1779.60 | 1774.71 | 1781.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1779.10 | 1775.59 | 1781.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:30:00 | 1787.00 | 1775.59 | 1781.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1779.90 | 1776.45 | 1780.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:00:00 | 1790.00 | 1779.16 | 1781.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1777.20 | 1778.77 | 1781.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 11:15:00 | 1774.90 | 1778.77 | 1781.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 1776.00 | 1777.00 | 1779.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:45:00 | 1767.40 | 1769.17 | 1775.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 1773.20 | 1762.43 | 1766.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 1764.10 | 1763.16 | 1766.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 1765.30 | 1763.16 | 1766.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 14:15:00 | 1772.10 | 1764.95 | 1766.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 15:00:00 | 1772.10 | 1764.95 | 1766.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1778.00 | 1767.56 | 1767.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1748.60 | 1767.56 | 1767.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1768.96 | 1767.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1791.30 | 1768.96 | 1767.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1814.90 | 1786.88 | 1777.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 1806.30 | 1819.99 | 1810.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1806.30 | 1819.99 | 1810.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1806.30 | 1819.99 | 1810.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 1806.30 | 1819.99 | 1810.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1809.00 | 1817.79 | 1810.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1818.00 | 1814.03 | 1810.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 09:45:00 | 1410.75 | 2023-05-23 09:15:00 | 1424.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-05-22 11:15:00 | 1411.00 | 2023-05-23 09:15:00 | 1424.55 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2023-05-30 14:15:00 | 1455.10 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2023-05-31 10:00:00 | 1455.00 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.08% |
| BUY | retest2 | 2023-05-31 12:30:00 | 1454.75 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2023-05-31 14:15:00 | 1454.70 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2023-06-01 11:45:00 | 1455.45 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2023-06-01 12:45:00 | 1456.50 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2023-06-02 09:15:00 | 1457.00 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2023-06-02 10:30:00 | 1455.25 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2023-06-09 09:15:00 | 1470.85 | 2023-06-09 11:15:00 | 1470.65 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2023-06-15 13:30:00 | 1480.85 | 2023-06-15 15:15:00 | 1471.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-06-20 13:00:00 | 1513.50 | 2023-06-22 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-06-22 13:00:00 | 1513.20 | 2023-06-22 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-06-23 15:00:00 | 1500.00 | 2023-06-27 12:15:00 | 1513.95 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2023-06-26 09:45:00 | 1498.50 | 2023-06-27 12:15:00 | 1513.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-06-26 11:00:00 | 1502.85 | 2023-06-27 12:15:00 | 1513.95 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-06-26 11:30:00 | 1503.00 | 2023-06-27 12:15:00 | 1513.95 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2023-06-30 09:15:00 | 1531.50 | 2023-07-10 10:15:00 | 1585.40 | STOP_HIT | 1.00 | 3.52% |
| BUY | retest2 | 2023-07-21 13:00:00 | 1642.50 | 2023-07-26 15:15:00 | 1626.30 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-07-24 10:30:00 | 1645.05 | 2023-07-26 15:15:00 | 1626.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-07-25 13:15:00 | 1644.95 | 2023-07-26 15:15:00 | 1626.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-08-08 11:15:00 | 1495.65 | 2023-08-08 15:15:00 | 1519.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2023-08-18 10:15:00 | 1459.75 | 2023-08-21 11:15:00 | 1479.95 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2023-08-18 12:30:00 | 1462.65 | 2023-08-21 11:15:00 | 1479.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2023-08-18 15:00:00 | 1461.15 | 2023-08-21 11:15:00 | 1479.95 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2023-08-28 13:30:00 | 1498.75 | 2023-08-31 13:15:00 | 1494.95 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-08-28 15:00:00 | 1501.00 | 2023-08-31 13:15:00 | 1494.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-08-31 09:30:00 | 1503.00 | 2023-08-31 13:15:00 | 1494.95 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-08-31 11:15:00 | 1499.00 | 2023-08-31 13:15:00 | 1494.95 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2023-09-04 15:15:00 | 1514.00 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-09-05 09:30:00 | 1516.00 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-09-05 10:45:00 | 1512.80 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-09-05 13:45:00 | 1513.30 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2023-09-06 11:15:00 | 1514.55 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-09-06 12:15:00 | 1514.05 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-09-06 15:00:00 | 1518.75 | 2023-09-13 09:15:00 | 1526.00 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2023-09-07 10:30:00 | 1515.10 | 2023-09-13 09:15:00 | 1526.00 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2023-09-12 11:15:00 | 1539.45 | 2023-09-13 11:15:00 | 1534.75 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-09-12 12:30:00 | 1536.30 | 2023-09-13 11:15:00 | 1534.75 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2023-09-18 09:15:00 | 1542.50 | 2023-09-18 09:15:00 | 1541.25 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2023-09-29 15:15:00 | 1539.00 | 2023-10-03 12:15:00 | 1553.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2023-10-03 11:00:00 | 1539.20 | 2023-10-03 12:15:00 | 1553.45 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2023-10-12 13:30:00 | 1636.35 | 2023-10-18 10:15:00 | 1631.40 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-10-12 14:00:00 | 1635.30 | 2023-10-18 10:15:00 | 1631.40 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2023-10-13 12:15:00 | 1635.00 | 2023-10-18 10:15:00 | 1631.40 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2023-10-16 09:45:00 | 1637.00 | 2023-10-18 12:15:00 | 1626.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1667.00 | 2023-10-18 12:15:00 | 1626.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2023-10-17 11:30:00 | 1651.90 | 2023-10-18 12:15:00 | 1626.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-10-18 10:00:00 | 1649.85 | 2023-10-18 12:15:00 | 1626.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2023-10-20 09:15:00 | 1624.35 | 2023-10-30 09:15:00 | 1543.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:30:00 | 1628.90 | 2023-10-30 09:15:00 | 1547.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 10:30:00 | 1624.45 | 2023-10-30 09:15:00 | 1543.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:15:00 | 1624.35 | 2023-10-31 09:15:00 | 1562.05 | STOP_HIT | 0.50 | 3.84% |
| SELL | retest2 | 2023-10-20 12:30:00 | 1628.90 | 2023-10-31 09:15:00 | 1562.05 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2023-10-23 10:30:00 | 1624.45 | 2023-10-31 09:15:00 | 1562.05 | STOP_HIT | 0.50 | 3.84% |
| BUY | retest2 | 2023-11-02 13:15:00 | 1579.20 | 2023-11-03 13:15:00 | 1556.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2023-11-02 13:45:00 | 1580.05 | 2023-11-03 13:15:00 | 1556.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2023-11-03 09:15:00 | 1586.45 | 2023-11-03 13:15:00 | 1556.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2023-11-07 10:15:00 | 1564.50 | 2023-11-07 14:15:00 | 1569.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2023-11-07 14:15:00 | 1564.05 | 2023-11-07 14:15:00 | 1569.40 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2023-11-10 10:45:00 | 1583.60 | 2023-11-20 11:15:00 | 1598.15 | STOP_HIT | 1.00 | 0.92% |
| BUY | retest2 | 2023-11-13 11:00:00 | 1582.50 | 2023-11-20 11:15:00 | 1598.15 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2023-11-13 14:45:00 | 1582.15 | 2023-11-20 11:15:00 | 1598.15 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2023-11-16 09:30:00 | 1581.65 | 2023-11-20 11:15:00 | 1598.15 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2023-11-28 09:15:00 | 1624.15 | 2023-12-13 09:15:00 | 1684.95 | STOP_HIT | 1.00 | 3.74% |
| SELL | retest2 | 2023-12-26 10:30:00 | 1669.00 | 2023-12-28 12:15:00 | 1677.25 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-12-28 10:45:00 | 1671.20 | 2023-12-28 12:15:00 | 1677.25 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2023-12-28 11:15:00 | 1670.30 | 2023-12-28 12:15:00 | 1677.25 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-01-04 09:15:00 | 1711.60 | 2024-01-09 14:15:00 | 1682.80 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-01-11 11:30:00 | 1677.10 | 2024-01-17 10:15:00 | 1593.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-11 11:30:00 | 1677.10 | 2024-01-19 09:15:00 | 1595.90 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2024-02-07 11:15:00 | 1597.05 | 2024-02-14 14:15:00 | 1578.95 | STOP_HIT | 1.00 | 1.13% |
| SELL | retest2 | 2024-02-07 14:45:00 | 1594.70 | 2024-02-14 14:15:00 | 1578.95 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-02-07 15:15:00 | 1594.85 | 2024-02-14 14:15:00 | 1578.95 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2024-02-27 12:30:00 | 1620.60 | 2024-02-27 13:15:00 | 1601.15 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-03-15 11:45:00 | 1562.35 | 2024-03-19 14:15:00 | 1581.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-03-15 13:15:00 | 1564.10 | 2024-03-19 14:15:00 | 1581.80 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-03-15 14:15:00 | 1562.85 | 2024-03-19 14:15:00 | 1581.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-03-15 15:15:00 | 1562.20 | 2024-03-19 15:15:00 | 1580.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-03-19 09:15:00 | 1558.75 | 2024-03-19 15:15:00 | 1580.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-03-19 09:45:00 | 1559.20 | 2024-03-19 15:15:00 | 1580.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-03-19 13:45:00 | 1560.60 | 2024-03-19 15:15:00 | 1580.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-03-22 14:00:00 | 1594.40 | 2024-03-22 14:15:00 | 1585.95 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-03-22 14:30:00 | 1594.80 | 2024-03-22 15:15:00 | 1584.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-03-26 09:15:00 | 1599.60 | 2024-03-27 12:15:00 | 1585.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-03-26 15:15:00 | 1595.00 | 2024-03-27 12:15:00 | 1585.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-04-02 11:15:00 | 1657.80 | 2024-04-15 09:15:00 | 1678.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-04-02 12:15:00 | 1659.60 | 2024-04-15 09:15:00 | 1678.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2024-04-02 14:30:00 | 1660.80 | 2024-04-15 09:15:00 | 1678.00 | STOP_HIT | 1.00 | 1.04% |
| BUY | retest2 | 2024-04-04 13:30:00 | 1657.90 | 2024-04-15 09:15:00 | 1678.00 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2024-04-22 13:00:00 | 1620.00 | 2024-04-22 14:15:00 | 1631.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-04-22 13:30:00 | 1620.00 | 2024-04-22 14:15:00 | 1631.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2024-04-24 09:15:00 | 1626.30 | 2024-04-26 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-04-24 09:45:00 | 1627.25 | 2024-04-26 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-04-25 09:30:00 | 1626.00 | 2024-04-26 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-04-25 10:00:00 | 1628.00 | 2024-04-26 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-04-25 12:15:00 | 1639.10 | 2024-04-26 09:15:00 | 1604.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1606.95 | 2024-05-16 14:15:00 | 1598.75 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-05-08 10:30:00 | 1610.45 | 2024-05-16 14:15:00 | 1598.75 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-05-08 13:00:00 | 1610.25 | 2024-05-16 14:15:00 | 1598.75 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-05-24 09:15:00 | 1616.25 | 2024-05-28 12:15:00 | 1593.45 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-05-24 10:30:00 | 1612.40 | 2024-05-28 12:15:00 | 1593.45 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-06-21 10:30:00 | 1582.95 | 2024-06-21 13:15:00 | 1593.60 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-06-21 12:30:00 | 1583.55 | 2024-06-21 13:15:00 | 1593.60 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-06-25 09:45:00 | 1578.35 | 2024-06-25 12:15:00 | 1589.05 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-06-25 10:15:00 | 1576.75 | 2024-06-25 12:15:00 | 1589.05 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-03 11:15:00 | 1581.25 | 2024-07-03 12:15:00 | 1591.45 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-10 15:15:00 | 1583.00 | 2024-07-23 11:15:00 | 1626.00 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2024-07-11 09:45:00 | 1585.45 | 2024-07-23 11:15:00 | 1626.00 | STOP_HIT | 1.00 | 2.56% |
| SELL | retest1 | 2024-08-06 10:30:00 | 1568.35 | 2024-08-12 10:15:00 | 1565.40 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest1 | 2024-08-06 11:00:00 | 1572.05 | 2024-08-12 10:15:00 | 1565.40 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest1 | 2024-08-06 13:45:00 | 1569.80 | 2024-08-12 10:15:00 | 1565.40 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest1 | 2024-08-07 11:15:00 | 1569.65 | 2024-08-12 10:15:00 | 1565.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-08-13 13:45:00 | 1545.40 | 2024-08-19 10:15:00 | 1549.70 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-09-09 12:30:00 | 1865.95 | 2024-09-10 09:15:00 | 1823.65 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-09-11 14:00:00 | 1833.10 | 2024-09-12 14:15:00 | 1853.40 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-09-12 09:30:00 | 1827.55 | 2024-09-12 14:15:00 | 1853.40 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-09-25 15:00:00 | 1928.50 | 2024-10-03 09:15:00 | 1945.50 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-10-11 09:30:00 | 1857.90 | 2024-10-11 13:15:00 | 1879.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-10-17 09:30:00 | 1840.35 | 2024-10-22 10:15:00 | 1748.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:30:00 | 1840.35 | 2024-10-23 09:15:00 | 1761.30 | STOP_HIT | 0.50 | 4.30% |
| BUY | retest2 | 2024-12-05 12:15:00 | 1630.40 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-12-05 15:00:00 | 1648.50 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-12-06 11:30:00 | 1633.40 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2024-12-09 11:15:00 | 1634.90 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2024-12-09 12:30:00 | 1644.35 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-12-09 13:15:00 | 1643.95 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1648.30 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-12-13 11:00:00 | 1643.70 | 2024-12-17 09:15:00 | 1641.60 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-01-10 11:00:00 | 1712.50 | 2025-01-13 10:15:00 | 1682.75 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-01-10 11:45:00 | 1707.30 | 2025-01-13 10:15:00 | 1682.75 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-01-10 13:15:00 | 1708.25 | 2025-01-13 10:15:00 | 1682.75 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-01-10 13:45:00 | 1708.40 | 2025-01-13 10:15:00 | 1682.75 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-01-22 13:45:00 | 1728.50 | 2025-01-27 09:15:00 | 1728.65 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-02-04 13:15:00 | 1785.00 | 2025-02-10 12:15:00 | 1781.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-02-06 12:30:00 | 1786.50 | 2025-02-10 12:15:00 | 1781.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-03-06 09:15:00 | 1793.00 | 2025-03-06 09:15:00 | 1814.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2025-03-10 09:15:00 | 1876.25 | 2025-03-10 14:15:00 | 1840.90 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-03-19 09:15:00 | 1859.20 | 2025-03-20 09:15:00 | 1828.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-03-19 11:30:00 | 1846.60 | 2025-03-20 09:15:00 | 1828.15 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-04-03 11:30:00 | 1912.90 | 2025-04-09 15:15:00 | 1900.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-04-04 09:30:00 | 1908.85 | 2025-04-09 15:15:00 | 1900.00 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-04-04 11:30:00 | 1912.60 | 2025-04-09 15:15:00 | 1900.00 | STOP_HIT | 1.00 | 0.66% |
| SELL | retest2 | 2025-04-04 14:00:00 | 1915.25 | 2025-04-09 15:15:00 | 1900.00 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-04-17 12:15:00 | 1986.80 | 2025-04-25 10:15:00 | 2033.80 | STOP_HIT | 1.00 | 2.37% |
| SELL | retest2 | 2025-05-05 11:30:00 | 2005.50 | 2025-05-05 13:15:00 | 2029.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-05-05 12:15:00 | 2005.90 | 2025-05-05 13:15:00 | 2029.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-05-13 15:15:00 | 2022.90 | 2025-05-15 09:15:00 | 2007.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-14 11:15:00 | 2023.20 | 2025-05-15 09:15:00 | 2007.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-05-27 11:15:00 | 2052.00 | 2025-05-27 14:15:00 | 2028.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1999.10 | 2025-06-02 14:15:00 | 2031.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-04 12:15:00 | 1991.00 | 2025-06-06 11:15:00 | 2011.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-06-04 13:00:00 | 1957.00 | 2025-06-06 11:15:00 | 2011.10 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-06-10 12:30:00 | 2011.50 | 2025-06-13 09:15:00 | 1997.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-06-11 10:00:00 | 2013.50 | 2025-06-13 10:15:00 | 1990.80 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-06-11 12:45:00 | 2009.00 | 2025-06-13 10:15:00 | 1990.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-06-11 14:15:00 | 2008.40 | 2025-06-13 10:15:00 | 1990.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-12 09:15:00 | 2043.00 | 2025-06-13 10:15:00 | 1990.80 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1969.70 | 2025-06-23 12:15:00 | 1993.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-06-20 14:15:00 | 1976.20 | 2025-06-23 12:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-06-20 15:15:00 | 1976.80 | 2025-06-23 12:15:00 | 1993.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-06-27 14:30:00 | 2048.40 | 2025-07-02 10:15:00 | 2019.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-06-30 09:15:00 | 2050.30 | 2025-07-02 10:15:00 | 2019.60 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-30 10:00:00 | 2049.00 | 2025-07-02 10:15:00 | 2019.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-30 11:45:00 | 2048.00 | 2025-07-02 10:15:00 | 2019.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-04 11:30:00 | 1993.80 | 2025-07-08 09:15:00 | 2007.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-07 12:45:00 | 1997.20 | 2025-07-08 09:15:00 | 2007.10 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-16 13:00:00 | 2032.00 | 2025-07-18 10:15:00 | 2015.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-17 09:45:00 | 2030.30 | 2025-07-18 10:15:00 | 2015.60 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-07-17 11:30:00 | 2032.00 | 2025-07-18 10:15:00 | 2015.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-17 12:15:00 | 2030.00 | 2025-07-18 10:15:00 | 2015.60 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-23 12:00:00 | 2055.00 | 2025-07-24 15:15:00 | 2030.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1990.60 | 2025-08-07 13:15:00 | 1891.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1990.60 | 2025-08-07 14:15:00 | 1912.80 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1948.40 | 2025-09-01 13:15:00 | 1939.30 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-09-19 11:30:00 | 2061.10 | 2025-09-22 09:15:00 | 2073.10 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-19 14:00:00 | 2061.00 | 2025-09-22 09:15:00 | 2073.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-30 11:45:00 | 2009.80 | 2025-10-06 10:15:00 | 2021.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-10-09 13:15:00 | 2015.40 | 2025-10-13 15:15:00 | 2023.90 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-09 14:00:00 | 2015.90 | 2025-10-13 15:15:00 | 2023.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-10 12:00:00 | 2015.20 | 2025-10-13 15:15:00 | 2023.90 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-10-15 09:15:00 | 2038.40 | 2025-10-28 11:15:00 | 2131.90 | STOP_HIT | 1.00 | 4.59% |
| SELL | retest2 | 2025-10-30 10:45:00 | 2121.10 | 2025-11-07 14:15:00 | 2100.30 | STOP_HIT | 1.00 | 0.98% |
| SELL | retest2 | 2025-10-30 11:45:00 | 2116.00 | 2025-11-07 14:15:00 | 2100.30 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-10-31 10:00:00 | 2119.20 | 2025-11-07 14:15:00 | 2100.30 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-12-01 09:15:00 | 2097.50 | 2025-12-01 11:15:00 | 2081.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-01 11:15:00 | 2097.00 | 2025-12-01 11:15:00 | 2081.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-04 12:30:00 | 2043.30 | 2025-12-05 09:15:00 | 2081.60 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-04 15:15:00 | 2048.50 | 2025-12-05 09:15:00 | 2081.60 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-12-19 11:15:00 | 2035.60 | 2025-12-19 15:15:00 | 2044.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-12-19 12:15:00 | 2032.60 | 2025-12-19 15:15:00 | 2044.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-22 12:15:00 | 2046.10 | 2025-12-24 12:15:00 | 2034.70 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-12-24 09:15:00 | 2058.10 | 2025-12-24 12:15:00 | 2034.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-31 09:15:00 | 2019.70 | 2025-12-31 11:15:00 | 2029.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-01-06 09:15:00 | 2065.10 | 2026-01-07 10:15:00 | 2031.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-06 13:30:00 | 2050.40 | 2026-01-07 10:15:00 | 2031.30 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1933.70 | 2026-01-29 12:15:00 | 1952.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-30 09:15:00 | 1930.50 | 2026-01-30 11:15:00 | 1947.10 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-30 10:15:00 | 1935.00 | 2026-01-30 11:15:00 | 1947.10 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-30 11:00:00 | 1931.90 | 2026-01-30 11:15:00 | 1947.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-02-12 10:45:00 | 2033.30 | 2026-02-13 12:15:00 | 2017.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-12 12:00:00 | 2033.00 | 2026-02-13 12:15:00 | 2017.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-02-12 12:30:00 | 2034.80 | 2026-02-13 12:15:00 | 2017.10 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-18 10:30:00 | 2062.90 | 2026-02-19 14:15:00 | 2033.60 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-02-18 15:00:00 | 2061.50 | 2026-02-19 14:15:00 | 2033.60 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1842.30 | 2026-03-13 12:15:00 | 1750.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:30:00 | 1842.30 | 2026-03-16 14:15:00 | 1775.00 | STOP_HIT | 0.50 | 3.65% |
| SELL | retest2 | 2026-03-24 10:30:00 | 1682.60 | 2026-03-25 09:15:00 | 1750.70 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-03-24 12:00:00 | 1681.10 | 2026-03-25 09:15:00 | 1750.70 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2026-04-01 10:30:00 | 1661.10 | 2026-04-06 14:15:00 | 1669.70 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-04-01 13:30:00 | 1658.50 | 2026-04-06 14:15:00 | 1669.70 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1660.30 | 2026-04-06 14:15:00 | 1669.70 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2026-04-06 13:45:00 | 1662.10 | 2026-04-06 14:15:00 | 1669.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-04-13 12:30:00 | 1782.00 | 2026-04-23 10:15:00 | 1796.20 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2026-04-29 11:15:00 | 1774.90 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-04-29 13:30:00 | 1776.00 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-30 09:45:00 | 1767.40 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2026-05-04 12:00:00 | 1773.20 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-05-05 09:15:00 | 1748.60 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 1.00 | -2.44% |
