# Lodha Developers Ltd. (LODHA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 61 |
| ALERT1 | 41 |
| ALERT2 | 40 |
| ALERT2_SKIP | 22 |
| ALERT3 | 105 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 37 / 31
- **Target hits / Stop hits / Partials:** 3 / 56 / 9
- **Avg / median % per leg:** 1.53% / 0.46%
- **Sum % (uncompounded):** 103.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 14 | 50.0% | 0 | 28 | 0 | 0.06% | 1.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 14 | 50.0% | 0 | 28 | 0 | 0.06% | 1.7% |
| SELL (all) | 40 | 23 | 57.5% | 3 | 28 | 9 | 2.55% | 102.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 23 | 57.5% | 3 | 28 | 9 | 2.55% | 102.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 37 | 54.4% | 3 | 56 | 9 | 1.53% | 103.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1306.00 | 1279.27 | 1278.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 1307.90 | 1288.98 | 1283.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1272.70 | 1285.73 | 1282.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 10:00:00 | 1272.70 | 1285.73 | 1282.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1271.60 | 1282.90 | 1281.67 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1268.50 | 1278.98 | 1280.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 13:15:00 | 1265.50 | 1276.28 | 1278.71 | Break + close below crossover candle low |

### Cycle 3 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1313.50 | 1282.99 | 1281.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 1323.00 | 1290.99 | 1284.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1408.80 | 1413.71 | 1392.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1393.50 | 1413.71 | 1392.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1394.50 | 1407.69 | 1396.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1394.50 | 1407.69 | 1396.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1380.60 | 1402.27 | 1394.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1380.60 | 1402.27 | 1394.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1385.00 | 1398.82 | 1394.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1404.10 | 1398.82 | 1394.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1437.80 | 1450.25 | 1450.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1437.80 | 1450.25 | 1450.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 10:15:00 | 1433.30 | 1446.86 | 1448.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1434.80 | 1431.04 | 1437.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 1434.80 | 1431.04 | 1437.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1439.40 | 1432.71 | 1437.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:30:00 | 1442.50 | 1432.71 | 1437.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1435.70 | 1433.31 | 1437.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1435.70 | 1433.31 | 1437.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1433.00 | 1433.25 | 1437.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 1430.20 | 1433.25 | 1437.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1457.40 | 1437.59 | 1438.51 | SL hit (close>static) qty=1.00 sl=1438.80 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 10:15:00 | 1455.10 | 1441.09 | 1440.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 11:15:00 | 1470.50 | 1446.97 | 1442.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 14:15:00 | 1447.10 | 1448.42 | 1444.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 15:00:00 | 1447.10 | 1448.42 | 1444.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 1447.50 | 1448.23 | 1444.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 1452.60 | 1448.23 | 1444.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1451.20 | 1448.83 | 1445.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 11:45:00 | 1453.80 | 1450.16 | 1446.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1463.00 | 1447.38 | 1446.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1459.70 | 1458.04 | 1453.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1471.00 | 1458.67 | 1453.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 13:15:00 | 1469.70 | 1489.56 | 1490.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 13:15:00 | 1469.70 | 1489.56 | 1490.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 14:15:00 | 1468.20 | 1485.28 | 1488.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1485.10 | 1482.80 | 1486.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1487.40 | 1482.80 | 1486.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1475.00 | 1476.78 | 1481.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1466.00 | 1476.78 | 1481.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 14:15:00 | 1470.10 | 1458.82 | 1458.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1470.10 | 1458.82 | 1458.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 1472.90 | 1461.64 | 1459.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1464.80 | 1466.90 | 1463.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:00:00 | 1464.80 | 1466.90 | 1463.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 1460.30 | 1465.58 | 1462.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:45:00 | 1460.40 | 1465.58 | 1462.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 1451.90 | 1462.85 | 1461.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 13:45:00 | 1447.40 | 1462.85 | 1461.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 1447.60 | 1459.80 | 1460.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 1440.30 | 1449.64 | 1454.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1450.60 | 1447.38 | 1451.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 1458.60 | 1447.38 | 1451.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1446.90 | 1447.29 | 1451.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 1447.40 | 1447.29 | 1451.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1456.80 | 1439.32 | 1444.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1451.00 | 1439.32 | 1444.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1457.70 | 1442.99 | 1445.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1455.00 | 1442.99 | 1445.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1454.10 | 1447.82 | 1447.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1481.60 | 1456.19 | 1451.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 1472.70 | 1475.20 | 1465.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 15:00:00 | 1472.70 | 1475.20 | 1465.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1473.00 | 1481.74 | 1474.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 1476.00 | 1481.74 | 1474.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1470.10 | 1479.41 | 1473.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1470.10 | 1479.41 | 1473.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1467.30 | 1476.99 | 1473.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1482.70 | 1476.99 | 1473.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1463.00 | 1474.56 | 1472.85 | SL hit (close<static) qty=1.00 sl=1467.30 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 11:15:00 | 1454.80 | 1470.61 | 1471.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 12:15:00 | 1453.10 | 1467.11 | 1469.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 14:15:00 | 1441.40 | 1439.54 | 1450.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1441.40 | 1439.54 | 1450.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1430.30 | 1437.75 | 1448.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-27 11:45:00 | 1422.40 | 1433.59 | 1444.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 1390.00 | 1375.26 | 1373.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 1390.00 | 1375.26 | 1373.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 1394.00 | 1379.01 | 1375.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 12:15:00 | 1407.40 | 1407.59 | 1398.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 13:00:00 | 1407.40 | 1407.59 | 1398.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1399.60 | 1405.99 | 1398.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 1398.70 | 1405.99 | 1398.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1400.10 | 1404.82 | 1398.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 10:00:00 | 1405.40 | 1403.70 | 1399.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1394.90 | 1401.94 | 1398.93 | SL hit (close<static) qty=1.00 sl=1396.70 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 1438.30 | 1440.49 | 1440.71 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 1442.90 | 1440.97 | 1440.91 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 09:15:00 | 1358.60 | 1425.16 | 1433.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 1339.00 | 1379.49 | 1406.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 1244.00 | 1234.75 | 1270.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 1249.10 | 1234.75 | 1270.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1248.00 | 1237.37 | 1244.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 1249.30 | 1237.37 | 1244.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1244.20 | 1238.74 | 1244.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:15:00 | 1233.40 | 1238.74 | 1244.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 14:15:00 | 1240.30 | 1225.46 | 1228.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 1242.30 | 1231.70 | 1230.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 1242.30 | 1231.70 | 1230.90 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1223.60 | 1230.08 | 1230.23 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 1237.00 | 1230.87 | 1230.47 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 1213.80 | 1227.35 | 1228.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 1206.60 | 1217.66 | 1222.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1222.20 | 1217.58 | 1221.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1222.20 | 1217.58 | 1221.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1221.50 | 1218.37 | 1221.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1226.60 | 1218.37 | 1221.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1215.70 | 1217.83 | 1220.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 1210.80 | 1217.29 | 1220.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:00:00 | 1210.60 | 1215.28 | 1218.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:45:00 | 1208.90 | 1212.83 | 1215.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1225.50 | 1218.75 | 1217.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1225.50 | 1218.75 | 1217.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 1227.30 | 1222.03 | 1219.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1219.30 | 1221.49 | 1219.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1219.30 | 1221.49 | 1219.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1215.00 | 1220.19 | 1219.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:30:00 | 1215.00 | 1220.19 | 1219.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1214.00 | 1218.95 | 1218.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 1214.00 | 1218.95 | 1218.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1215.60 | 1218.28 | 1218.50 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1223.10 | 1219.24 | 1218.91 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 10:15:00 | 1215.60 | 1218.52 | 1218.61 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 1221.30 | 1219.07 | 1218.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 1223.30 | 1219.92 | 1219.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1240.20 | 1245.88 | 1239.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1228.50 | 1245.88 | 1239.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1247.10 | 1246.13 | 1240.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:00:00 | 1250.00 | 1246.90 | 1241.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 12:45:00 | 1250.80 | 1247.38 | 1241.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:30:00 | 1249.20 | 1247.61 | 1242.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 14:15:00 | 1251.20 | 1247.61 | 1242.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1283.30 | 1293.35 | 1283.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1274.90 | 1293.35 | 1283.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1276.10 | 1289.90 | 1282.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1276.10 | 1289.90 | 1282.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1274.30 | 1286.78 | 1281.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:45:00 | 1277.70 | 1286.78 | 1281.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-22 14:15:00 | 1260.50 | 1277.41 | 1278.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 1260.50 | 1277.41 | 1278.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1246.00 | 1264.94 | 1270.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1202.00 | 1202.00 | 1211.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:45:00 | 1202.40 | 1202.00 | 1211.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1213.10 | 1205.64 | 1211.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 1214.80 | 1205.64 | 1211.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1208.50 | 1206.21 | 1210.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1209.30 | 1206.21 | 1210.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1199.20 | 1200.00 | 1204.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:45:00 | 1203.50 | 1200.00 | 1204.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1202.10 | 1200.23 | 1203.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1210.30 | 1200.23 | 1203.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1215.00 | 1203.19 | 1204.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:30:00 | 1207.30 | 1204.19 | 1204.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:15:00 | 1200.40 | 1204.19 | 1204.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 1189.20 | 1176.34 | 1175.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 1189.20 | 1176.34 | 1175.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1205.00 | 1194.90 | 1187.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 1197.80 | 1197.87 | 1191.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 1197.80 | 1197.87 | 1191.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1200.80 | 1203.31 | 1200.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1200.80 | 1203.31 | 1200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1208.70 | 1204.39 | 1201.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:15:00 | 1212.90 | 1204.39 | 1201.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 1212.20 | 1216.00 | 1212.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 1203.10 | 1209.90 | 1210.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 1203.10 | 1209.90 | 1210.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 1199.00 | 1205.93 | 1208.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1153.70 | 1150.18 | 1160.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 1153.70 | 1150.18 | 1160.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1155.00 | 1151.29 | 1156.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1145.70 | 1151.29 | 1156.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1144.80 | 1149.99 | 1155.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1142.00 | 1149.37 | 1154.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 1141.30 | 1147.44 | 1153.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 1140.50 | 1140.85 | 1147.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 12:45:00 | 1142.20 | 1142.02 | 1146.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1141.20 | 1141.86 | 1145.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 1141.20 | 1141.86 | 1145.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1125.20 | 1138.19 | 1143.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1118.00 | 1133.97 | 1140.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:45:00 | 1117.00 | 1115.13 | 1121.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 13:15:00 | 1144.10 | 1122.89 | 1122.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 1144.10 | 1122.89 | 1122.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 15:15:00 | 1145.00 | 1130.61 | 1126.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1126.40 | 1131.73 | 1129.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1126.40 | 1131.73 | 1129.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1128.00 | 1130.99 | 1129.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1120.10 | 1130.67 | 1129.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1122.50 | 1129.04 | 1128.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:45:00 | 1124.90 | 1129.04 | 1128.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 11:15:00 | 1122.40 | 1127.71 | 1127.92 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1146.20 | 1131.37 | 1129.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1150.80 | 1137.84 | 1132.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1146.60 | 1147.96 | 1141.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:00:00 | 1146.60 | 1147.96 | 1141.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1147.40 | 1151.38 | 1146.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1147.40 | 1151.38 | 1146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1140.30 | 1149.16 | 1146.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1141.90 | 1149.16 | 1146.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1140.80 | 1147.49 | 1145.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 1138.10 | 1147.49 | 1145.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 1138.60 | 1143.96 | 1144.37 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1173.20 | 1149.80 | 1146.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 10:15:00 | 1182.00 | 1156.24 | 1150.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 1185.70 | 1186.83 | 1177.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 1185.70 | 1186.83 | 1177.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1182.80 | 1185.17 | 1180.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:45:00 | 1177.70 | 1185.17 | 1180.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1189.20 | 1185.98 | 1181.01 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1178.10 | 1183.06 | 1183.27 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 09:15:00 | 1191.50 | 1184.75 | 1184.02 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 14:15:00 | 1172.60 | 1182.75 | 1183.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 14:15:00 | 1167.40 | 1174.67 | 1177.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1176.80 | 1175.10 | 1177.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1177.00 | 1175.10 | 1177.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1178.20 | 1175.72 | 1177.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 1182.30 | 1175.72 | 1177.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1180.90 | 1176.76 | 1178.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 1185.30 | 1176.76 | 1178.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1185.00 | 1178.40 | 1178.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:45:00 | 1185.50 | 1178.40 | 1178.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1184.50 | 1179.62 | 1179.30 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1173.30 | 1178.68 | 1179.01 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1202.80 | 1179.86 | 1178.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1223.80 | 1198.66 | 1189.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1226.10 | 1227.60 | 1216.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 1226.10 | 1227.60 | 1216.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1222.00 | 1224.80 | 1217.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:45:00 | 1232.80 | 1226.23 | 1220.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 1236.00 | 1228.18 | 1221.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:30:00 | 1234.20 | 1228.98 | 1222.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1201.40 | 1223.18 | 1221.09 | SL hit (close<static) qty=1.00 sl=1213.90 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 1211.80 | 1219.16 | 1220.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 14:15:00 | 1206.90 | 1215.94 | 1218.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1217.90 | 1211.03 | 1214.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1217.90 | 1211.03 | 1214.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1219.00 | 1212.62 | 1214.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 1217.50 | 1212.62 | 1214.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1231.30 | 1218.27 | 1217.05 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 1215.00 | 1220.11 | 1220.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1205.60 | 1217.20 | 1218.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1218.70 | 1213.06 | 1215.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 1219.30 | 1213.06 | 1215.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1219.70 | 1214.39 | 1215.57 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1220.30 | 1217.02 | 1216.61 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 1205.90 | 1214.73 | 1215.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1191.90 | 1206.18 | 1211.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1203.20 | 1200.27 | 1206.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1203.20 | 1200.27 | 1206.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1202.60 | 1201.70 | 1205.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1198.10 | 1201.16 | 1204.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:00:00 | 1197.20 | 1199.52 | 1203.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1184.50 | 1199.16 | 1201.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 1138.19 | 1148.18 | 1153.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 11:15:00 | 1137.34 | 1148.18 | 1153.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-01 15:15:00 | 1125.27 | 1137.74 | 1146.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1112.90 | 1107.50 | 1116.83 | SL hit (close>ema200) qty=0.50 sl=1107.50 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1085.80 | 1082.22 | 1081.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1093.10 | 1087.30 | 1084.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1080.90 | 1086.18 | 1084.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1079.20 | 1086.18 | 1084.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1087.30 | 1086.40 | 1084.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 11:15:00 | 1093.90 | 1086.40 | 1084.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 1078.50 | 1086.13 | 1086.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1078.50 | 1086.13 | 1086.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1076.80 | 1084.27 | 1085.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1072.00 | 1066.39 | 1072.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1072.00 | 1066.39 | 1072.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1074.60 | 1068.03 | 1072.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 1075.60 | 1068.03 | 1072.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1070.50 | 1068.52 | 1072.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 1068.00 | 1068.52 | 1072.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1079.50 | 1070.65 | 1072.42 | SL hit (close>static) qty=1.00 sl=1074.60 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1081.60 | 1074.89 | 1074.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1090.60 | 1078.03 | 1075.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1087.90 | 1090.85 | 1085.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 1084.70 | 1090.85 | 1085.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1084.70 | 1089.62 | 1085.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1088.70 | 1085.86 | 1085.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 1081.40 | 1084.97 | 1084.82 | SL hit (close<static) qty=1.00 sl=1084.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 1080.50 | 1084.07 | 1084.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 1079.20 | 1082.60 | 1083.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1082.20 | 1082.04 | 1083.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1083.30 | 1082.04 | 1083.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1076.90 | 1081.01 | 1082.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 1081.20 | 1081.01 | 1082.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1080.50 | 1080.33 | 1081.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:45:00 | 1081.90 | 1080.33 | 1081.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 1075.00 | 1079.27 | 1081.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 14:00:00 | 1074.00 | 1078.21 | 1080.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:00:00 | 1072.00 | 1075.87 | 1078.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 11:15:00 | 1070.50 | 1065.60 | 1065.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 11:15:00 | 1070.50 | 1065.60 | 1065.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 09:15:00 | 1080.00 | 1071.64 | 1068.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 1073.70 | 1074.27 | 1070.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:45:00 | 1073.10 | 1074.27 | 1070.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1110.60 | 1111.42 | 1100.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 1124.50 | 1110.39 | 1103.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:45:00 | 1119.20 | 1113.22 | 1106.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1119.60 | 1111.87 | 1108.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1099.70 | 1110.38 | 1108.46 | SL hit (close<static) qty=1.00 sl=1100.50 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1092.80 | 1106.87 | 1107.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 1091.50 | 1102.21 | 1104.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1070.80 | 1068.72 | 1078.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1070.00 | 1068.72 | 1078.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1077.30 | 1071.35 | 1077.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1079.00 | 1071.35 | 1077.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1074.00 | 1071.88 | 1077.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1068.00 | 1071.88 | 1077.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1069.00 | 1071.30 | 1076.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1058.60 | 1065.94 | 1071.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 1079.20 | 1068.57 | 1068.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1079.20 | 1068.57 | 1068.26 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1057.00 | 1066.86 | 1068.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 1048.50 | 1063.19 | 1066.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 910.00 | 901.33 | 922.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 910.00 | 901.33 | 922.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 919.30 | 906.55 | 921.14 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 09:15:00 | 949.70 | 924.96 | 924.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 954.70 | 937.18 | 930.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 955.35 | 961.46 | 951.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 952.85 | 961.46 | 951.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 951.75 | 959.76 | 952.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 951.75 | 959.76 | 952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 961.50 | 960.11 | 952.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 947.50 | 960.11 | 952.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 950.35 | 958.16 | 952.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:00:00 | 950.35 | 958.16 | 952.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 915.00 | 949.53 | 949.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 915.00 | 949.53 | 949.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 921.55 | 943.93 | 946.75 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 961.85 | 949.59 | 948.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 963.50 | 952.37 | 949.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1040.75 | 1041.46 | 1019.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1040.75 | 1041.46 | 1019.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1029.25 | 1041.39 | 1031.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1029.25 | 1041.39 | 1031.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1042.15 | 1041.54 | 1032.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:45:00 | 1046.35 | 1043.24 | 1035.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1050.05 | 1079.10 | 1081.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1050.05 | 1079.10 | 1081.16 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1089.10 | 1078.65 | 1078.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 1096.00 | 1084.89 | 1081.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 13:15:00 | 1092.55 | 1093.15 | 1088.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 14:00:00 | 1092.55 | 1093.15 | 1088.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1088.85 | 1093.04 | 1089.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 1088.85 | 1093.04 | 1089.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1093.00 | 1093.03 | 1089.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 11:45:00 | 1096.60 | 1093.99 | 1090.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 10:15:00 | 1086.50 | 1096.90 | 1094.25 | SL hit (close<static) qty=1.00 sl=1089.15 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1072.45 | 1092.01 | 1092.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 12:15:00 | 1068.50 | 1087.31 | 1090.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1078.00 | 1075.97 | 1081.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 1081.30 | 1075.97 | 1081.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1076.55 | 1076.09 | 1081.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 1080.15 | 1076.09 | 1081.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1080.70 | 1077.01 | 1081.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 1081.45 | 1077.01 | 1081.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1071.00 | 1075.81 | 1080.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 1068.10 | 1075.81 | 1080.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 1066.90 | 1072.27 | 1077.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1067.15 | 1071.98 | 1075.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1014.69 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1013.56 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:15:00 | 1013.79 | 1025.90 | 1040.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 961.29 | 992.28 | 1007.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 860.60 | 854.46 | 854.26 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 824.80 | 848.38 | 851.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 817.00 | 830.79 | 840.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 752.80 | 735.92 | 754.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 750.15 | 735.92 | 754.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 744.90 | 737.72 | 753.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 741.90 | 738.94 | 752.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 740.20 | 738.94 | 752.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 735.10 | 740.11 | 750.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 10:15:00 | 704.80 | 726.95 | 740.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 11:15:00 | 703.19 | 722.43 | 737.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-27 14:15:00 | 698.35 | 711.15 | 728.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 690.00 | 685.78 | 701.64 | SL hit (close>ema200) qty=0.50 sl=685.78 alert=retest2 |

### Cycle 59 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 693.50 | 690.46 | 690.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 701.70 | 692.71 | 691.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 816.00 | 818.83 | 797.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:00:00 | 824.80 | 820.02 | 799.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 842.00 | 821.77 | 808.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 12:15:00 | 864.25 | 872.74 | 873.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 864.25 | 872.74 | 873.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 855.70 | 867.81 | 870.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 853.50 | 847.98 | 855.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 10:45:00 | 852.55 | 847.98 | 855.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 854.25 | 849.23 | 855.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 862.40 | 849.23 | 855.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 865.00 | 852.38 | 856.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:45:00 | 864.00 | 852.38 | 856.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 868.45 | 855.60 | 857.18 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 880.00 | 860.48 | 859.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 885.30 | 871.08 | 864.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 901.40 | 904.56 | 893.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 893.55 | 900.56 | 895.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 893.55 | 900.56 | 895.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 906.30 | 900.56 | 895.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:30:00 | 1284.40 | 2025-05-12 13:15:00 | 1306.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-12 11:15:00 | 1294.50 | 2025-05-12 13:15:00 | 1306.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1404.10 | 2025-05-30 09:15:00 | 1437.80 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-06-02 15:15:00 | 1430.20 | 2025-06-03 09:15:00 | 1457.40 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-06-04 09:15:00 | 1452.60 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.18% |
| BUY | retest2 | 2025-06-04 10:00:00 | 1451.20 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.27% |
| BUY | retest2 | 2025-06-04 11:45:00 | 1453.80 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1463.00 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1471.00 | 2025-06-10 13:15:00 | 1469.70 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-06-12 09:15:00 | 1466.00 | 2025-06-16 14:15:00 | 1470.10 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1482.70 | 2025-06-25 10:15:00 | 1463.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-27 11:45:00 | 1422.40 | 2025-07-09 10:15:00 | 1390.00 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2025-07-14 10:00:00 | 1405.40 | 2025-07-14 10:15:00 | 1394.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-07-14 12:30:00 | 1404.00 | 2025-07-22 12:15:00 | 1438.30 | STOP_HIT | 1.00 | 2.44% |
| BUY | retest2 | 2025-07-14 13:15:00 | 1406.80 | 2025-07-22 12:15:00 | 1438.30 | STOP_HIT | 1.00 | 2.24% |
| SELL | retest2 | 2025-07-31 14:15:00 | 1233.40 | 2025-08-04 15:15:00 | 1242.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-04 14:15:00 | 1240.30 | 2025-08-04 15:15:00 | 1242.30 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-08 10:30:00 | 1210.80 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-08-08 14:00:00 | 1210.60 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-11 11:45:00 | 1208.90 | 2025-08-11 15:15:00 | 1225.50 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1250.00 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-08-19 12:45:00 | 1250.80 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-08-19 13:30:00 | 1249.20 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-08-19 14:15:00 | 1251.20 | 2025-08-22 14:15:00 | 1260.50 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-09-04 10:30:00 | 1207.30 | 2025-09-15 13:15:00 | 1189.20 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-09-04 11:15:00 | 1200.40 | 2025-09-15 13:15:00 | 1189.20 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-09-19 13:15:00 | 1212.90 | 2025-09-23 12:15:00 | 1203.10 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-23 10:15:00 | 1212.20 | 2025-09-23 12:15:00 | 1203.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1142.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-09-30 11:30:00 | 1141.30 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-10-01 10:00:00 | 1140.50 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-10-01 12:45:00 | 1142.20 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1118.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-07 09:45:00 | 1117.00 | 2025-10-07 13:15:00 | 1144.10 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-11-06 12:45:00 | 1232.80 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-11-06 14:00:00 | 1236.00 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-11-06 14:30:00 | 1234.20 | 2025-11-07 09:15:00 | 1201.40 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1198.10 | 2025-12-01 11:15:00 | 1138.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1197.20 | 2025-12-01 11:15:00 | 1137.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-01 15:15:00 | 1125.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1198.10 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 7.11% |
| SELL | retest2 | 2025-11-20 11:00:00 | 1197.20 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 7.04% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1184.50 | 2025-12-04 11:15:00 | 1112.90 | STOP_HIT | 0.50 | 6.04% |
| BUY | retest2 | 2025-12-15 11:15:00 | 1093.90 | 2025-12-16 12:15:00 | 1078.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-18 14:15:00 | 1068.00 | 2025-12-19 09:15:00 | 1079.50 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1088.70 | 2025-12-24 09:15:00 | 1081.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1074.00 | 2026-01-01 11:15:00 | 1070.50 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-12-29 10:00:00 | 1072.00 | 2026-01-01 11:15:00 | 1070.50 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2026-01-07 09:15:00 | 1124.50 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2026-01-07 10:45:00 | 1119.20 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-01-08 09:15:00 | 1119.60 | 2026-01-08 11:15:00 | 1099.70 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1058.60 | 2026-01-16 11:15:00 | 1079.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-06 13:45:00 | 1046.35 | 2026-02-13 09:15:00 | 1050.05 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-18 11:45:00 | 1096.60 | 2026-02-19 10:15:00 | 1086.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1068.10 | 2026-02-26 10:15:00 | 1014.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1066.90 | 2026-02-26 10:15:00 | 1013.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1067.15 | 2026-02-26 10:15:00 | 1013.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-20 15:15:00 | 1068.10 | 2026-03-02 09:15:00 | 961.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 11:30:00 | 1066.90 | 2026-03-02 09:15:00 | 960.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1067.15 | 2026-03-02 09:15:00 | 960.44 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 741.90 | 2026-03-27 10:15:00 | 704.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 740.20 | 2026-03-27 11:15:00 | 703.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 735.10 | 2026-03-27 14:15:00 | 698.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 741.90 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 7.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 740.20 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 6.78% |
| SELL | retest2 | 2026-03-25 14:15:00 | 735.10 | 2026-04-01 09:15:00 | 690.00 | STOP_HIT | 0.50 | 6.14% |
| BUY | retest2 | 2026-04-13 11:00:00 | 824.80 | 2026-04-23 12:15:00 | 864.25 | STOP_HIT | 1.00 | 4.78% |
| BUY | retest2 | 2026-04-15 09:15:00 | 842.00 | 2026-04-23 12:15:00 | 864.25 | STOP_HIT | 1.00 | 2.64% |
