# INDUSINDBK (INDUSINDBK)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2026-05-08 15:15:00 (3717 bars)
- **Last close:** 948.45
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 100 |
| ALERT2 | 100 |
| ALERT2_SKIP | 50 |
| ALERT3 | 276 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 135 |
| PARTIAL | 20 |
| TARGET_HIT | 4 |
| STOP_HIT | 130 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 154 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 66 / 88
- **Target hits / Stop hits / Partials:** 4 / 130 / 20
- **Avg / median % per leg:** 0.89% / -0.47%
- **Sum % (uncompounded):** 136.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 13 | 23.2% | 0 | 56 | 0 | -0.42% | -23.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 56 | 13 | 23.2% | 0 | 56 | 0 | -0.42% | -23.6% |
| SELL (all) | 98 | 53 | 54.1% | 4 | 74 | 20 | 1.63% | 160.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 98 | 53 | 54.1% | 4 | 74 | 20 | 1.63% | 160.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 154 | 66 | 42.9% | 4 | 130 | 20 | 0.89% | 136.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 1420.95 | 1415.14 | 1414.40 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 10:15:00 | 1401.35 | 1412.53 | 1413.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 1387.00 | 1404.90 | 1409.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 1410.00 | 1405.92 | 1409.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 1410.00 | 1405.92 | 1409.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1410.00 | 1405.92 | 1409.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1410.00 | 1405.92 | 1409.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1410.00 | 1406.73 | 1409.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1397.75 | 1406.73 | 1409.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 11:15:00 | 1406.65 | 1407.05 | 1409.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 12:00:00 | 1404.05 | 1406.45 | 1408.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 1407.90 | 1406.74 | 1408.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 1408.75 | 1407.14 | 1408.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:30:00 | 1408.25 | 1407.14 | 1408.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 1414.55 | 1408.62 | 1409.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 1414.55 | 1408.62 | 1409.33 | SL hit (close>static) qty=1.00 sl=1412.40 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 1415.00 | 1409.90 | 1409.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 09:15:00 | 1418.50 | 1411.62 | 1410.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1411.45 | 1413.16 | 1411.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1411.45 | 1413.16 | 1411.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1411.45 | 1413.16 | 1411.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1411.45 | 1413.16 | 1411.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1409.45 | 1412.42 | 1411.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 1409.45 | 1412.42 | 1411.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1410.55 | 1412.04 | 1411.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 1410.55 | 1412.04 | 1411.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1414.20 | 1412.47 | 1411.69 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 14:15:00 | 1404.10 | 1410.73 | 1411.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 15:15:00 | 1402.80 | 1409.15 | 1410.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 1407.40 | 1404.36 | 1406.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 1407.40 | 1404.36 | 1406.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1407.40 | 1404.36 | 1406.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1407.40 | 1404.36 | 1406.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1412.00 | 1405.89 | 1407.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 1415.00 | 1405.89 | 1407.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 1419.90 | 1408.69 | 1408.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 11:15:00 | 1429.15 | 1414.64 | 1411.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 1428.35 | 1429.98 | 1421.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 1428.35 | 1429.98 | 1421.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1450.00 | 1459.91 | 1453.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:15:00 | 1441.65 | 1459.91 | 1453.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1441.30 | 1456.18 | 1452.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:30:00 | 1443.70 | 1456.18 | 1452.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1449.15 | 1454.78 | 1452.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:45:00 | 1453.05 | 1455.41 | 1452.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 15:15:00 | 1452.00 | 1455.78 | 1455.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 15:15:00 | 1452.00 | 1455.02 | 1455.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 15:15:00 | 1452.00 | 1455.02 | 1455.10 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 14:15:00 | 1463.60 | 1455.83 | 1455.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1517.90 | 1469.40 | 1461.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1471.00 | 1504.28 | 1488.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1471.00 | 1504.28 | 1488.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1471.00 | 1504.28 | 1488.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:00:00 | 1471.00 | 1504.28 | 1488.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1390.15 | 1481.45 | 1479.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1390.15 | 1481.45 | 1479.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1386.15 | 1462.39 | 1471.05 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 14:15:00 | 1501.00 | 1457.53 | 1454.93 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 11:15:00 | 1479.75 | 1483.92 | 1484.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 13:15:00 | 1477.55 | 1481.72 | 1483.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 14:15:00 | 1485.65 | 1482.51 | 1483.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 14:15:00 | 1485.65 | 1482.51 | 1483.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1485.65 | 1482.51 | 1483.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:00:00 | 1485.65 | 1482.51 | 1483.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1482.60 | 1482.53 | 1483.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 09:15:00 | 1488.20 | 1482.53 | 1483.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1487.05 | 1483.43 | 1483.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 11:00:00 | 1483.05 | 1483.36 | 1483.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 11:15:00 | 1485.00 | 1483.68 | 1483.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 11:15:00 | 1485.00 | 1483.68 | 1483.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 12:15:00 | 1489.00 | 1484.75 | 1484.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-14 13:15:00 | 1499.15 | 1503.46 | 1496.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-14 14:00:00 | 1499.15 | 1503.46 | 1496.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1504.90 | 1503.75 | 1497.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:15:00 | 1498.50 | 1503.75 | 1497.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 15:15:00 | 1498.50 | 1502.70 | 1497.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:30:00 | 1507.60 | 1502.96 | 1498.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 10:30:00 | 1509.00 | 1503.60 | 1498.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 11:00:00 | 1506.15 | 1503.60 | 1498.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 1482.85 | 1518.41 | 1521.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 1482.85 | 1518.41 | 1521.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 12:15:00 | 1475.75 | 1486.75 | 1492.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 09:15:00 | 1439.50 | 1438.11 | 1451.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 09:45:00 | 1437.80 | 1438.11 | 1451.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 1449.00 | 1439.97 | 1448.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 1449.00 | 1439.97 | 1448.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 1452.75 | 1442.53 | 1449.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:00:00 | 1452.75 | 1442.53 | 1449.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1455.45 | 1445.11 | 1449.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 1455.45 | 1445.11 | 1449.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1453.00 | 1446.69 | 1450.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 1450.70 | 1446.69 | 1450.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1444.95 | 1443.83 | 1447.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:00:00 | 1444.95 | 1443.83 | 1447.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1444.30 | 1443.93 | 1446.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 1440.55 | 1443.93 | 1446.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1430.05 | 1441.20 | 1445.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 1424.10 | 1432.64 | 1435.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:30:00 | 1424.60 | 1429.97 | 1433.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 15:15:00 | 1425.00 | 1427.44 | 1431.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 09:30:00 | 1423.40 | 1426.20 | 1429.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1423.10 | 1422.91 | 1426.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:00:00 | 1423.10 | 1422.91 | 1426.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1430.40 | 1424.41 | 1427.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 1430.40 | 1424.41 | 1427.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1432.00 | 1425.93 | 1427.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:15:00 | 1437.35 | 1425.93 | 1427.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 1444.40 | 1431.50 | 1430.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 1444.40 | 1431.50 | 1430.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 1454.60 | 1444.34 | 1440.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1441.05 | 1448.85 | 1445.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 1441.05 | 1448.85 | 1445.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1441.05 | 1448.85 | 1445.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 1441.05 | 1448.85 | 1445.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1435.10 | 1446.10 | 1444.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:00:00 | 1435.10 | 1446.10 | 1444.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1433.10 | 1441.57 | 1442.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 1422.10 | 1434.46 | 1438.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 1426.90 | 1426.37 | 1432.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 14:00:00 | 1426.90 | 1426.37 | 1432.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 1394.00 | 1388.44 | 1395.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 1394.00 | 1388.44 | 1395.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 1396.20 | 1389.99 | 1395.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:30:00 | 1400.90 | 1389.99 | 1395.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 1400.75 | 1392.15 | 1395.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:00:00 | 1400.75 | 1392.15 | 1395.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 1404.40 | 1394.60 | 1396.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 1404.40 | 1394.60 | 1396.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 09:15:00 | 1424.15 | 1402.01 | 1399.58 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 1397.65 | 1419.92 | 1421.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1380.75 | 1403.75 | 1411.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 1390.30 | 1390.28 | 1400.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 14:45:00 | 1389.55 | 1390.28 | 1400.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1390.50 | 1389.80 | 1398.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 1388.40 | 1389.80 | 1398.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:30:00 | 1387.95 | 1391.84 | 1397.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 1388.05 | 1391.84 | 1397.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 09:30:00 | 1383.30 | 1384.24 | 1392.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1347.65 | 1348.39 | 1358.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1344.55 | 1348.39 | 1358.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:45:00 | 1346.90 | 1348.38 | 1357.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:15:00 | 1346.45 | 1348.38 | 1357.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 1346.80 | 1348.06 | 1356.67 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 10:15:00 | 1351.50 | 1348.49 | 1353.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:45:00 | 1355.90 | 1348.49 | 1353.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 11:15:00 | 1351.40 | 1349.07 | 1352.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 1366.75 | 1355.35 | 1354.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 1366.75 | 1355.35 | 1354.45 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 1345.00 | 1353.93 | 1354.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 1341.40 | 1349.45 | 1351.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1345.85 | 1344.75 | 1347.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1345.85 | 1344.75 | 1347.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1345.85 | 1344.75 | 1347.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:45:00 | 1340.65 | 1343.71 | 1347.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 1362.45 | 1349.39 | 1348.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 1362.45 | 1349.39 | 1348.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 1372.70 | 1357.64 | 1354.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 09:15:00 | 1370.00 | 1375.31 | 1367.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 09:15:00 | 1370.00 | 1375.31 | 1367.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1370.00 | 1375.31 | 1367.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:45:00 | 1369.95 | 1375.31 | 1367.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 1381.00 | 1384.63 | 1379.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:15:00 | 1380.90 | 1384.63 | 1379.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1380.40 | 1383.79 | 1379.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 14:00:00 | 1385.65 | 1382.88 | 1380.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 1385.95 | 1387.09 | 1384.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 11:15:00 | 1386.20 | 1385.21 | 1384.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 12:15:00 | 1379.90 | 1383.54 | 1383.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 1379.90 | 1383.54 | 1383.67 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 14:15:00 | 1384.15 | 1383.76 | 1383.75 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 1380.05 | 1383.02 | 1383.41 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 1403.85 | 1387.19 | 1385.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 10:15:00 | 1411.35 | 1392.02 | 1387.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 13:15:00 | 1410.35 | 1413.77 | 1405.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 14:00:00 | 1410.35 | 1413.77 | 1405.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1439.90 | 1441.88 | 1435.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1432.80 | 1441.88 | 1435.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1429.75 | 1439.45 | 1435.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:15:00 | 1436.40 | 1435.61 | 1434.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:45:00 | 1437.10 | 1436.09 | 1434.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 1436.05 | 1436.08 | 1434.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1436.65 | 1436.25 | 1434.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 10:15:00 | 1432.15 | 1435.43 | 1434.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 10:45:00 | 1432.10 | 1435.43 | 1434.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1436.00 | 1435.54 | 1434.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-05 12:15:00 | 1427.60 | 1433.96 | 1434.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1427.60 | 1433.96 | 1434.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 1425.40 | 1432.24 | 1433.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1418.40 | 1414.92 | 1420.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:00:00 | 1418.40 | 1414.92 | 1420.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1416.45 | 1415.23 | 1420.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 1420.10 | 1415.23 | 1420.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1422.70 | 1417.46 | 1420.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:30:00 | 1420.40 | 1417.46 | 1420.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1427.65 | 1419.50 | 1421.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1428.80 | 1419.50 | 1421.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1429.70 | 1422.89 | 1422.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 12:15:00 | 1436.30 | 1428.23 | 1425.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 11:15:00 | 1430.80 | 1432.81 | 1429.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 11:15:00 | 1430.80 | 1432.81 | 1429.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 1430.80 | 1432.81 | 1429.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:30:00 | 1434.65 | 1432.81 | 1429.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1424.75 | 1431.20 | 1428.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 1424.40 | 1431.20 | 1428.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1422.85 | 1429.53 | 1428.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1422.85 | 1429.53 | 1428.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 1420.20 | 1426.45 | 1427.10 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 1432.60 | 1427.29 | 1427.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 1436.90 | 1429.22 | 1428.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1464.35 | 1466.40 | 1458.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 1464.35 | 1466.40 | 1458.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 1470.00 | 1469.63 | 1463.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:15:00 | 1472.20 | 1469.63 | 1463.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1474.20 | 1470.54 | 1464.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 11:30:00 | 1478.70 | 1473.52 | 1467.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:45:00 | 1483.15 | 1476.44 | 1469.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1492.35 | 1476.84 | 1471.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:45:00 | 1478.15 | 1482.97 | 1478.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1486.30 | 1483.63 | 1479.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 1466.50 | 1479.00 | 1479.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 10:15:00 | 1466.50 | 1479.00 | 1479.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 11:15:00 | 1456.20 | 1474.44 | 1477.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 1442.85 | 1442.23 | 1450.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:15:00 | 1447.25 | 1442.23 | 1450.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1451.40 | 1444.70 | 1448.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:15:00 | 1453.00 | 1444.70 | 1448.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1453.00 | 1446.36 | 1448.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1465.20 | 1446.36 | 1448.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1470.70 | 1451.23 | 1450.72 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 1440.75 | 1453.59 | 1454.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 1431.10 | 1443.69 | 1448.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 1408.50 | 1401.09 | 1413.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 1409.00 | 1401.09 | 1413.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1365.35 | 1363.84 | 1371.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 1370.25 | 1363.84 | 1371.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 1354.00 | 1353.56 | 1358.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:15:00 | 1361.80 | 1353.56 | 1358.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 1361.80 | 1355.21 | 1359.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:45:00 | 1367.85 | 1356.79 | 1359.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1347.35 | 1354.90 | 1358.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1343.70 | 1354.90 | 1358.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 1364.50 | 1356.95 | 1356.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1364.50 | 1356.95 | 1356.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 1366.45 | 1358.85 | 1357.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1360.00 | 1362.51 | 1359.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1360.00 | 1362.51 | 1359.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1360.00 | 1362.51 | 1359.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1360.00 | 1362.51 | 1359.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1353.85 | 1360.78 | 1359.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:00:00 | 1353.85 | 1360.78 | 1359.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1354.35 | 1359.49 | 1358.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 1351.75 | 1359.49 | 1358.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 1357.65 | 1358.54 | 1358.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:15:00 | 1357.00 | 1358.54 | 1358.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1360.05 | 1358.84 | 1358.48 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 1344.70 | 1356.35 | 1357.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 1343.20 | 1353.72 | 1356.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 1348.15 | 1345.20 | 1348.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 1348.15 | 1345.20 | 1348.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 1348.15 | 1345.20 | 1348.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 1348.15 | 1345.20 | 1348.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 1346.45 | 1345.45 | 1348.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 1336.90 | 1345.45 | 1348.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 1343.90 | 1346.20 | 1348.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 12:45:00 | 1344.50 | 1343.23 | 1346.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 13:15:00 | 1352.60 | 1345.10 | 1346.83 | SL hit (close>static) qty=1.00 sl=1350.50 alert=retest2 |

### Cycle 33 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 1084.10 | 1066.89 | 1064.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 1090.25 | 1079.64 | 1072.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 09:15:00 | 1075.20 | 1080.30 | 1074.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1075.20 | 1080.30 | 1074.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1075.20 | 1080.30 | 1074.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 1075.20 | 1080.30 | 1074.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1069.45 | 1078.13 | 1073.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:30:00 | 1071.00 | 1078.13 | 1073.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 1074.20 | 1077.35 | 1073.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 15:00:00 | 1078.60 | 1076.32 | 1074.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 09:15:00 | 1064.00 | 1074.48 | 1073.64 | SL hit (close<static) qty=1.00 sl=1068.50 alert=retest2 |

### Cycle 34 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1063.80 | 1072.34 | 1072.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 13:15:00 | 1054.55 | 1066.24 | 1069.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 10:15:00 | 1058.70 | 1054.99 | 1059.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 1058.70 | 1054.99 | 1059.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 1061.45 | 1056.28 | 1059.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 11:45:00 | 1062.90 | 1056.28 | 1059.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 1064.75 | 1057.98 | 1060.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 1064.15 | 1057.98 | 1060.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 1056.50 | 1057.68 | 1059.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:15:00 | 1055.80 | 1058.79 | 1060.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:45:00 | 1054.95 | 1058.57 | 1059.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 10:00:00 | 1054.20 | 1058.03 | 1059.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1003.01 | 1022.95 | 1035.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1002.20 | 1022.95 | 1035.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 1001.49 | 1022.95 | 1035.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 1022.90 | 1012.03 | 1022.13 | SL hit (close>ema200) qty=0.50 sl=1012.03 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 1018.10 | 1002.61 | 1000.57 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 999.40 | 1004.53 | 1004.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 996.90 | 1002.03 | 1003.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 996.10 | 994.46 | 997.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 996.10 | 994.46 | 997.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 996.10 | 994.46 | 997.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 996.10 | 994.46 | 997.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 991.20 | 987.46 | 991.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 991.20 | 987.46 | 991.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1005.00 | 991.53 | 992.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 1005.00 | 991.53 | 992.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1005.25 | 994.28 | 993.96 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 993.35 | 996.31 | 996.34 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 11:15:00 | 1000.35 | 997.12 | 996.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 12:15:00 | 1002.05 | 998.10 | 997.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 15:15:00 | 997.80 | 998.35 | 997.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 15:15:00 | 997.80 | 998.35 | 997.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 997.80 | 998.35 | 997.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 996.35 | 998.35 | 997.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 989.50 | 996.58 | 996.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 09:15:00 | 980.70 | 989.80 | 993.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 10:15:00 | 986.05 | 985.46 | 988.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 10:30:00 | 984.50 | 985.46 | 988.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 982.35 | 983.68 | 986.15 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 12:15:00 | 997.40 | 988.03 | 987.60 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 15:15:00 | 984.30 | 986.96 | 987.21 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 09:15:00 | 989.40 | 987.45 | 987.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 10:15:00 | 1000.00 | 989.96 | 988.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 977.50 | 992.16 | 991.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 977.50 | 992.16 | 991.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 977.50 | 992.16 | 991.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 976.70 | 992.16 | 991.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 970.90 | 987.91 | 989.21 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1001.20 | 989.50 | 989.01 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 985.05 | 991.77 | 991.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 978.15 | 989.04 | 990.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 964.55 | 963.33 | 970.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 964.55 | 963.33 | 970.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 934.25 | 936.92 | 942.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 932.30 | 936.92 | 942.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 14:30:00 | 930.30 | 935.16 | 939.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 958.00 | 939.20 | 940.51 | SL hit (close>static) qty=1.00 sl=945.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 967.90 | 944.94 | 943.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 975.45 | 965.76 | 960.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 996.35 | 996.76 | 985.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 996.35 | 996.76 | 985.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 996.55 | 996.60 | 987.59 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 966.30 | 981.96 | 982.79 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 12:15:00 | 989.95 | 983.09 | 982.37 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 974.60 | 981.05 | 981.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 11:15:00 | 970.65 | 978.97 | 980.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 980.00 | 978.10 | 980.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 980.00 | 978.10 | 980.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 980.00 | 978.10 | 980.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 980.00 | 978.10 | 980.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 983.15 | 979.11 | 980.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:45:00 | 982.50 | 979.11 | 980.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 980.40 | 979.37 | 980.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:45:00 | 980.55 | 979.31 | 980.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 971.00 | 977.65 | 979.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:30:00 | 977.90 | 977.65 | 979.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 977.40 | 977.66 | 979.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 958.15 | 979.34 | 979.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 973.25 | 952.86 | 957.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 10:45:00 | 972.95 | 956.88 | 958.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 13:15:00 | 969.50 | 960.65 | 959.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 13:15:00 | 969.50 | 960.65 | 959.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 971.40 | 964.67 | 962.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 12:15:00 | 964.85 | 965.36 | 962.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-15 13:00:00 | 964.85 | 965.36 | 962.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 958.95 | 964.08 | 962.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:00:00 | 958.95 | 964.08 | 962.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 961.90 | 963.64 | 962.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-16 09:15:00 | 977.60 | 963.41 | 962.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:30:00 | 966.25 | 973.48 | 970.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 10:15:00 | 974.45 | 973.48 | 970.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 11:15:00 | 966.95 | 971.72 | 969.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 967.15 | 970.81 | 969.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 12:30:00 | 970.75 | 970.20 | 969.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 14:30:00 | 973.50 | 969.46 | 969.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 09:15:00 | 963.50 | 968.94 | 969.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 963.50 | 968.94 | 969.02 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 972.40 | 969.64 | 969.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 975.05 | 970.72 | 969.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 13:15:00 | 971.35 | 971.90 | 970.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 13:15:00 | 971.35 | 971.90 | 970.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 971.35 | 971.90 | 970.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:00:00 | 971.35 | 971.90 | 970.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 971.15 | 971.75 | 970.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 971.10 | 971.75 | 970.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 970.00 | 971.40 | 970.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 969.15 | 971.40 | 970.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 965.10 | 970.14 | 970.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 965.10 | 970.14 | 970.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 963.80 | 968.87 | 969.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 957.95 | 962.57 | 965.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 972.45 | 961.38 | 963.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 972.45 | 961.38 | 963.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 972.45 | 961.38 | 963.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 972.45 | 961.38 | 963.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 970.30 | 963.17 | 964.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 968.10 | 963.17 | 964.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 09:15:00 | 977.00 | 965.93 | 965.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 10:15:00 | 982.40 | 969.23 | 967.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-23 14:15:00 | 970.65 | 972.55 | 969.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-23 14:45:00 | 971.00 | 972.55 | 969.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 15:15:00 | 969.70 | 971.98 | 969.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:15:00 | 965.85 | 971.98 | 969.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 955.45 | 968.68 | 968.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 955.45 | 968.68 | 968.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 959.85 | 966.91 | 967.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 950.25 | 961.23 | 964.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 944.50 | 936.12 | 945.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 10:15:00 | 944.50 | 936.12 | 945.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 944.50 | 936.12 | 945.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 944.50 | 936.12 | 945.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 941.90 | 937.28 | 944.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:30:00 | 943.85 | 937.28 | 944.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 946.60 | 939.14 | 944.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:45:00 | 946.05 | 939.14 | 944.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 941.70 | 939.65 | 944.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 937.70 | 938.59 | 943.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 09:45:00 | 936.00 | 937.21 | 942.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-29 14:15:00 | 947.60 | 939.71 | 941.36 | SL hit (close>static) qty=1.00 sl=947.55 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 956.80 | 944.94 | 943.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 965.60 | 953.78 | 948.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 1063.60 | 1064.54 | 1051.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 13:45:00 | 1063.45 | 1064.54 | 1051.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1064.50 | 1071.37 | 1064.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1064.50 | 1071.37 | 1064.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1067.40 | 1070.58 | 1064.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 12:15:00 | 1073.10 | 1070.56 | 1065.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:00:00 | 1071.35 | 1070.72 | 1065.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 13:30:00 | 1072.10 | 1069.71 | 1065.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-11 09:30:00 | 1072.40 | 1068.08 | 1065.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 1056.30 | 1065.72 | 1065.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-11 10:15:00 | 1056.30 | 1065.72 | 1065.01 | SL hit (close<static) qty=1.00 sl=1062.70 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 12:15:00 | 1055.05 | 1063.02 | 1063.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 13:15:00 | 1054.55 | 1061.33 | 1063.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 12:15:00 | 1046.05 | 1045.95 | 1053.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 12:30:00 | 1042.35 | 1045.95 | 1053.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 1048.55 | 1043.66 | 1048.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 1048.55 | 1043.66 | 1048.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 1050.50 | 1045.03 | 1049.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:45:00 | 1054.85 | 1045.03 | 1049.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 12:15:00 | 1044.10 | 1044.84 | 1048.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 1041.20 | 1044.36 | 1048.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1031.80 | 1042.69 | 1046.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 15:15:00 | 1050.30 | 1041.68 | 1040.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 15:15:00 | 1050.30 | 1041.68 | 1040.53 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 1029.05 | 1039.15 | 1039.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 13:15:00 | 1027.95 | 1034.18 | 1036.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 09:15:00 | 1037.35 | 1031.75 | 1034.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 1037.35 | 1031.75 | 1034.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1037.35 | 1031.75 | 1034.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1037.35 | 1031.75 | 1034.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1029.55 | 1031.31 | 1034.32 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 1045.00 | 1036.58 | 1035.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 1048.75 | 1041.92 | 1038.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 12:15:00 | 1042.15 | 1043.47 | 1040.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 12:15:00 | 1042.15 | 1043.47 | 1040.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 1042.15 | 1043.47 | 1040.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:00:00 | 1042.15 | 1043.47 | 1040.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1046.35 | 1044.04 | 1041.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:30:00 | 1039.75 | 1044.04 | 1041.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1043.00 | 1043.96 | 1041.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 1029.70 | 1043.96 | 1041.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1029.90 | 1041.15 | 1040.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:45:00 | 1030.50 | 1041.15 | 1040.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 1026.35 | 1038.19 | 1039.44 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 10:15:00 | 1053.25 | 1039.22 | 1037.47 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 997.65 | 1035.09 | 1037.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 15:15:00 | 972.30 | 994.68 | 1013.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 13:15:00 | 977.30 | 976.96 | 995.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 992.50 | 982.24 | 993.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 992.50 | 982.24 | 993.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 992.50 | 982.24 | 993.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 994.25 | 984.65 | 993.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:30:00 | 992.65 | 984.65 | 993.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 11:15:00 | 987.05 | 985.13 | 993.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 984.30 | 985.13 | 993.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 14:45:00 | 985.50 | 986.15 | 991.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-05 10:00:00 | 984.80 | 986.17 | 990.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 12:15:00 | 935.08 | 954.43 | 967.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 12:15:00 | 936.22 | 954.43 | 967.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 12:15:00 | 935.56 | 954.43 | 967.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-10 09:15:00 | 885.87 | 938.05 | 954.83 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 673.30 | 660.01 | 659.34 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 646.00 | 657.49 | 658.99 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 09:15:00 | 682.40 | 660.36 | 659.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 687.50 | 677.92 | 670.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 685.00 | 701.77 | 694.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 685.00 | 701.77 | 694.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 685.00 | 701.77 | 694.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 685.00 | 701.77 | 694.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 685.40 | 698.50 | 694.07 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 678.40 | 691.36 | 691.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 655.45 | 681.11 | 686.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 677.15 | 673.11 | 679.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-07 15:00:00 | 677.15 | 673.11 | 679.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 680.50 | 675.45 | 679.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 676.80 | 676.56 | 679.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 678.65 | 676.56 | 679.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:30:00 | 679.60 | 678.07 | 679.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 678.35 | 679.60 | 680.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 673.60 | 678.40 | 679.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:00:00 | 672.00 | 677.12 | 678.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 682.60 | 679.65 | 679.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 682.60 | 679.65 | 679.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 691.15 | 682.73 | 681.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 09:15:00 | 804.50 | 817.49 | 798.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 804.50 | 817.49 | 798.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 804.50 | 817.49 | 798.48 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 10:15:00 | 784.25 | 793.10 | 793.47 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 12:15:00 | 799.35 | 793.93 | 793.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 822.00 | 799.89 | 796.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 11:15:00 | 816.85 | 819.70 | 812.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-25 12:00:00 | 816.85 | 819.70 | 812.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 817.50 | 832.04 | 829.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 15:00:00 | 837.90 | 832.67 | 830.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:15:00 | 862.50 | 832.74 | 830.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 829.10 | 840.00 | 841.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 829.10 | 840.00 | 841.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 10:15:00 | 818.45 | 830.75 | 835.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 13:15:00 | 832.90 | 830.54 | 834.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 14:00:00 | 832.90 | 830.54 | 834.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 830.35 | 830.50 | 833.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 830.35 | 830.50 | 833.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 827.85 | 829.90 | 833.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 11:30:00 | 825.60 | 828.66 | 831.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 14:15:00 | 825.00 | 827.77 | 830.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 15:15:00 | 818.60 | 827.82 | 830.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:30:00 | 819.35 | 819.89 | 823.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 10:15:00 | 784.32 | 794.14 | 806.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 10:15:00 | 783.75 | 794.14 | 806.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 10:15:00 | 777.67 | 794.14 | 806.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-13 10:15:00 | 778.38 | 794.14 | 806.29 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-14 11:15:00 | 782.05 | 777.02 | 788.63 | SL hit (close>ema200) qty=0.50 sl=777.02 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 783.35 | 777.19 | 777.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 799.05 | 785.06 | 781.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 12:15:00 | 785.00 | 786.29 | 782.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 13:00:00 | 785.00 | 786.29 | 782.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 782.75 | 785.93 | 783.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 781.80 | 785.93 | 783.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 782.30 | 785.20 | 783.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 772.35 | 785.20 | 783.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 10:15:00 | 773.95 | 781.15 | 781.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 763.35 | 777.59 | 780.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 795.30 | 776.53 | 777.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 795.30 | 776.53 | 777.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 795.30 | 776.53 | 777.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 795.30 | 776.53 | 777.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 762.85 | 773.79 | 776.51 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 15:15:00 | 783.50 | 776.94 | 776.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 789.50 | 780.90 | 778.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 813.15 | 814.87 | 805.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 09:45:00 | 814.90 | 814.87 | 805.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 807.65 | 813.36 | 807.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 807.65 | 813.36 | 807.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 806.20 | 811.93 | 807.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:30:00 | 806.55 | 811.93 | 807.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 805.00 | 810.54 | 807.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 807.20 | 810.54 | 807.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 15:15:00 | 812.05 | 813.79 | 813.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 812.05 | 813.79 | 813.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 806.05 | 812.00 | 813.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 806.00 | 805.39 | 808.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 806.00 | 805.39 | 808.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 806.00 | 805.39 | 808.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 805.70 | 805.39 | 808.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 809.35 | 806.18 | 808.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 809.35 | 806.18 | 808.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 814.25 | 807.80 | 809.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:45:00 | 813.35 | 807.80 | 809.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 814.95 | 810.10 | 810.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 15:15:00 | 815.20 | 811.12 | 810.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 810.60 | 811.02 | 810.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 810.60 | 811.02 | 810.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 810.60 | 811.02 | 810.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 810.60 | 811.02 | 810.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 805.00 | 809.81 | 810.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 802.40 | 807.06 | 808.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 813.00 | 807.14 | 808.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 813.00 | 807.14 | 808.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 813.00 | 807.14 | 808.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 813.00 | 807.14 | 808.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 810.95 | 807.91 | 808.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 812.45 | 807.91 | 808.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 841.65 | 814.59 | 811.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 853.10 | 837.12 | 827.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 838.75 | 844.17 | 839.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 838.75 | 844.17 | 839.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 838.75 | 844.17 | 839.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 838.75 | 844.17 | 839.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 836.40 | 842.62 | 838.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 15:00:00 | 836.40 | 842.62 | 838.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 836.65 | 841.42 | 838.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 836.35 | 841.42 | 838.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 833.90 | 839.28 | 838.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 832.75 | 839.28 | 838.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 831.35 | 836.51 | 837.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 827.00 | 833.70 | 835.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 822.65 | 820.05 | 824.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:00:00 | 822.65 | 820.05 | 824.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 822.90 | 820.62 | 824.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 824.10 | 820.62 | 824.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 821.10 | 820.72 | 824.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:30:00 | 821.10 | 820.72 | 824.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 851.75 | 820.66 | 820.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:00:00 | 851.75 | 820.66 | 820.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 838.50 | 824.22 | 822.30 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 10:15:00 | 831.15 | 836.54 | 837.25 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 840.40 | 834.38 | 833.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 866.20 | 841.96 | 837.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 850.30 | 871.95 | 866.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 850.30 | 871.95 | 866.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 850.30 | 871.95 | 866.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 850.30 | 871.95 | 866.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 859.00 | 869.36 | 865.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 863.15 | 868.12 | 865.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 13:00:00 | 865.00 | 867.49 | 865.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 858.00 | 864.03 | 864.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 858.00 | 864.03 | 864.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 853.65 | 859.21 | 860.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 857.30 | 857.26 | 859.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 857.30 | 857.26 | 859.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 857.30 | 857.26 | 859.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 859.20 | 857.26 | 859.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 861.50 | 857.67 | 859.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 843.60 | 848.85 | 852.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:15:00 | 844.80 | 847.14 | 850.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:30:00 | 844.30 | 847.55 | 850.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 854.00 | 851.74 | 851.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 10:15:00 | 854.00 | 851.74 | 851.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 11:15:00 | 856.70 | 852.73 | 851.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 15:15:00 | 878.25 | 879.80 | 875.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 09:15:00 | 884.10 | 879.80 | 875.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 876.00 | 879.04 | 875.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 871.15 | 879.04 | 875.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 866.80 | 876.59 | 874.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 866.80 | 876.59 | 874.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 869.60 | 875.19 | 874.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:15:00 | 871.00 | 875.19 | 874.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 13:15:00 | 866.80 | 872.14 | 872.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 866.80 | 872.14 | 872.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 15:15:00 | 863.00 | 869.31 | 871.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 10:15:00 | 869.40 | 868.88 | 870.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 11:00:00 | 869.40 | 868.88 | 870.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 870.65 | 869.23 | 870.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 874.00 | 869.23 | 870.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 866.95 | 868.78 | 870.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:30:00 | 868.50 | 868.78 | 870.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 872.50 | 869.52 | 870.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:45:00 | 870.40 | 869.52 | 870.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 869.65 | 869.55 | 870.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:15:00 | 874.00 | 869.55 | 870.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 874.00 | 870.44 | 870.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 845.00 | 870.44 | 870.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 802.75 | 815.02 | 826.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 806.20 | 805.38 | 815.97 | SL hit (close>ema200) qty=0.50 sl=805.38 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 802.95 | 798.03 | 797.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 824.95 | 803.41 | 800.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 808.45 | 813.04 | 808.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 808.45 | 813.04 | 808.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 808.45 | 813.04 | 808.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 808.45 | 813.04 | 808.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 798.10 | 810.05 | 807.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 798.10 | 810.05 | 807.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 804.20 | 808.88 | 806.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 797.75 | 808.88 | 806.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 805.90 | 808.41 | 807.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 805.90 | 808.41 | 807.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 803.20 | 807.37 | 806.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 803.20 | 807.37 | 806.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 802.15 | 806.32 | 806.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 805.25 | 806.32 | 806.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 798.90 | 804.84 | 805.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 794.60 | 801.49 | 803.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 807.35 | 800.41 | 802.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 807.35 | 800.41 | 802.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 807.35 | 800.41 | 802.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 807.35 | 800.41 | 802.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 808.60 | 802.05 | 803.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 806.30 | 802.05 | 803.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-14 13:15:00 | 765.98 | 773.10 | 777.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 790.25 | 775.62 | 777.51 | SL hit (close>ema200) qty=0.50 sl=775.62 alert=retest2 |

### Cycle 89 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 792.65 | 779.02 | 778.89 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 779.30 | 783.51 | 783.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 13:15:00 | 779.00 | 782.61 | 783.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 768.80 | 765.67 | 769.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 11:00:00 | 768.80 | 765.67 | 769.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 771.20 | 766.77 | 769.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 771.20 | 766.77 | 769.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 773.50 | 768.12 | 770.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:30:00 | 774.00 | 768.12 | 770.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 772.00 | 769.72 | 770.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 766.05 | 769.72 | 770.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 754.90 | 750.88 | 750.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 754.90 | 750.88 | 750.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 12:15:00 | 756.85 | 752.91 | 751.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 750.65 | 752.46 | 751.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 750.65 | 752.46 | 751.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 750.65 | 752.46 | 751.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 750.65 | 752.46 | 751.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 751.55 | 752.28 | 751.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 757.00 | 752.12 | 751.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 753.05 | 757.48 | 757.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 753.05 | 757.48 | 757.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 752.20 | 756.12 | 756.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 756.90 | 755.57 | 756.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 756.90 | 755.57 | 756.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 756.90 | 755.57 | 756.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 757.50 | 755.57 | 756.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 756.95 | 755.84 | 756.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 755.40 | 755.84 | 756.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 758.25 | 756.32 | 756.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 756.45 | 756.32 | 756.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 749.15 | 752.40 | 754.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 14:30:00 | 754.40 | 752.40 | 754.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 749.70 | 752.11 | 753.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 748.15 | 750.51 | 752.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 12:30:00 | 749.05 | 749.69 | 750.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 13:30:00 | 749.05 | 750.69 | 750.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:00:00 | 748.20 | 750.19 | 750.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 749.25 | 749.81 | 750.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 10:15:00 | 748.20 | 749.81 | 750.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 745.15 | 740.38 | 739.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 745.15 | 740.38 | 739.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 747.65 | 742.41 | 741.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 11:15:00 | 742.00 | 742.70 | 741.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 742.00 | 742.70 | 741.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 742.30 | 742.62 | 741.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 742.00 | 742.62 | 741.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 739.15 | 741.92 | 741.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 739.15 | 741.92 | 741.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 734.95 | 740.53 | 740.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 731.90 | 737.77 | 739.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 743.25 | 738.51 | 739.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 743.25 | 738.51 | 739.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 743.25 | 738.51 | 739.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 743.25 | 738.51 | 739.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 12:15:00 | 749.05 | 740.61 | 740.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 13:15:00 | 755.25 | 743.54 | 741.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 11:15:00 | 746.55 | 749.16 | 745.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 11:15:00 | 746.55 | 749.16 | 745.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 746.55 | 749.16 | 745.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 746.55 | 749.16 | 745.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 745.70 | 748.47 | 745.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 745.35 | 748.47 | 745.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 743.15 | 747.41 | 745.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 744.00 | 747.41 | 745.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 740.60 | 746.05 | 745.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 740.60 | 746.05 | 745.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 10:15:00 | 741.25 | 744.04 | 744.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 11:15:00 | 736.15 | 742.46 | 743.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 742.15 | 741.97 | 743.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 13:45:00 | 741.60 | 741.97 | 743.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 740.30 | 741.64 | 742.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 733.80 | 741.43 | 742.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 734.70 | 729.74 | 729.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 734.70 | 729.74 | 729.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 739.20 | 732.82 | 731.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 10:15:00 | 742.80 | 745.54 | 741.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 10:15:00 | 742.80 | 745.54 | 741.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 742.80 | 745.54 | 741.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 742.80 | 745.54 | 741.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 741.35 | 744.71 | 741.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:45:00 | 741.00 | 744.71 | 741.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 738.80 | 743.52 | 741.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 738.80 | 743.52 | 741.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 738.75 | 742.57 | 740.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:15:00 | 740.80 | 742.57 | 740.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 741.30 | 741.79 | 740.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 742.40 | 744.30 | 744.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 742.40 | 744.30 | 744.35 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 748.30 | 745.10 | 744.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 754.45 | 749.19 | 747.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 13:15:00 | 760.25 | 760.98 | 756.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 14:00:00 | 760.25 | 760.98 | 756.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 753.45 | 758.96 | 756.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 753.45 | 758.96 | 756.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 747.95 | 756.75 | 756.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 747.95 | 756.75 | 756.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 743.70 | 754.14 | 754.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 12:15:00 | 742.10 | 747.06 | 750.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 09:15:00 | 740.15 | 740.13 | 743.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 740.15 | 740.13 | 743.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 751.80 | 742.47 | 744.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 751.80 | 742.47 | 744.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 755.95 | 745.16 | 745.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 768.20 | 753.85 | 749.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 15:15:00 | 758.90 | 759.14 | 754.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 760.10 | 759.07 | 754.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 763.95 | 759.72 | 755.82 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 754.85 | 756.57 | 756.65 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 759.60 | 756.52 | 756.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 760.60 | 757.34 | 756.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 798.45 | 801.97 | 794.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:00:00 | 798.45 | 801.97 | 794.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 800.85 | 801.30 | 796.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 805.95 | 801.29 | 797.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 15:15:00 | 794.50 | 798.36 | 796.87 | SL hit (close<static) qty=1.00 sl=795.70 alert=retest2 |

### Cycle 104 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 790.95 | 796.16 | 796.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 15:15:00 | 787.10 | 793.51 | 795.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 789.05 | 788.53 | 792.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 789.05 | 788.53 | 792.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 788.90 | 786.81 | 789.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 786.60 | 786.81 | 789.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 790.70 | 787.59 | 789.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:30:00 | 790.45 | 787.59 | 789.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 790.45 | 788.16 | 789.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 791.20 | 788.16 | 789.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 791.10 | 788.75 | 789.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 791.00 | 788.75 | 789.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 796.95 | 790.39 | 790.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 798.50 | 790.39 | 790.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 797.05 | 791.72 | 791.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 799.30 | 793.24 | 791.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 797.00 | 797.19 | 794.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-11 09:15:00 | 807.00 | 797.19 | 794.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 820.50 | 801.85 | 797.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 824.55 | 801.85 | 797.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:30:00 | 821.35 | 813.67 | 805.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 824.40 | 816.32 | 807.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 14:15:00 | 847.55 | 851.65 | 852.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 14:15:00 | 847.55 | 851.65 | 852.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 842.45 | 849.19 | 850.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 842.05 | 835.56 | 839.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 842.05 | 835.56 | 839.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 842.05 | 835.56 | 839.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 842.05 | 835.56 | 839.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 844.45 | 837.34 | 840.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 844.45 | 837.34 | 840.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 843.75 | 838.62 | 840.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:15:00 | 846.40 | 838.62 | 840.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 854.95 | 843.71 | 842.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 856.65 | 847.11 | 844.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 12:15:00 | 847.90 | 848.30 | 845.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 847.90 | 848.30 | 845.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 844.95 | 847.63 | 845.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 844.95 | 847.63 | 845.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 836.95 | 845.49 | 844.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 836.95 | 845.49 | 844.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 835.00 | 843.40 | 844.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 831.05 | 839.30 | 841.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 13:15:00 | 840.80 | 837.94 | 840.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 13:15:00 | 840.80 | 837.94 | 840.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 840.80 | 837.94 | 840.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 840.80 | 837.94 | 840.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 840.40 | 838.44 | 840.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:15:00 | 842.50 | 838.44 | 840.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 842.50 | 839.25 | 840.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 846.45 | 839.25 | 840.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 855.00 | 842.40 | 841.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 11:15:00 | 861.60 | 852.39 | 848.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 11:15:00 | 855.25 | 855.38 | 852.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:00:00 | 855.25 | 855.38 | 852.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 855.00 | 855.09 | 852.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:15:00 | 857.45 | 855.09 | 852.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:00:00 | 856.50 | 856.23 | 853.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 848.15 | 854.20 | 853.33 | SL hit (close<static) qty=1.00 sl=851.50 alert=retest2 |

### Cycle 110 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 846.85 | 851.91 | 852.40 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 860.50 | 850.63 | 850.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 871.10 | 854.73 | 852.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 859.60 | 860.73 | 857.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 859.60 | 860.73 | 857.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 859.60 | 860.73 | 857.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 859.60 | 860.73 | 857.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 869.65 | 862.22 | 858.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:30:00 | 873.30 | 863.59 | 859.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 10:15:00 | 853.65 | 862.77 | 860.76 | SL hit (close<static) qty=1.00 sl=857.50 alert=retest2 |

### Cycle 112 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 841.95 | 856.18 | 857.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 837.90 | 852.52 | 856.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 847.10 | 846.10 | 850.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:45:00 | 847.20 | 846.10 | 850.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 845.90 | 845.53 | 848.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 839.25 | 843.75 | 847.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:00:00 | 837.15 | 841.65 | 845.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 840.00 | 840.43 | 845.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 835.45 | 837.08 | 841.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 841.50 | 837.96 | 841.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 841.50 | 837.96 | 841.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 835.15 | 837.40 | 841.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 833.20 | 836.75 | 840.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 15:15:00 | 833.85 | 836.75 | 840.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 848.10 | 838.55 | 840.39 | SL hit (close>static) qty=1.00 sl=842.65 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 846.30 | 841.39 | 841.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 849.45 | 844.58 | 843.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 844.35 | 847.01 | 845.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 844.35 | 847.01 | 845.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 844.35 | 847.01 | 845.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 844.35 | 847.01 | 845.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 845.60 | 846.72 | 845.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 843.80 | 846.72 | 845.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 843.05 | 845.99 | 844.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 843.05 | 845.99 | 844.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 843.65 | 845.52 | 844.86 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 838.10 | 843.89 | 844.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 830.30 | 838.74 | 841.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 09:15:00 | 838.50 | 837.53 | 840.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:00:00 | 838.50 | 837.53 | 840.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 838.45 | 837.72 | 840.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 840.10 | 837.72 | 840.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 840.10 | 838.19 | 840.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 840.10 | 838.19 | 840.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 838.00 | 838.15 | 839.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 836.90 | 838.15 | 839.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 846.35 | 837.73 | 838.05 | SL hit (close>static) qty=1.00 sl=841.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 844.05 | 838.99 | 838.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 847.85 | 841.09 | 839.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 852.55 | 853.06 | 847.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 849.90 | 853.06 | 847.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 850.15 | 852.29 | 848.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 849.85 | 852.29 | 848.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 849.15 | 851.66 | 848.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 849.05 | 851.66 | 848.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 851.00 | 851.53 | 849.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 848.50 | 851.53 | 849.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 846.40 | 850.51 | 848.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 847.75 | 850.51 | 848.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 850.30 | 850.46 | 848.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 852.55 | 850.46 | 848.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:45:00 | 855.60 | 850.30 | 849.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 852.00 | 851.34 | 850.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 838.50 | 848.28 | 849.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 838.50 | 848.28 | 849.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 837.20 | 841.61 | 845.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 13:15:00 | 839.50 | 839.01 | 842.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 13:45:00 | 839.95 | 839.01 | 842.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 841.60 | 839.53 | 842.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 842.40 | 839.53 | 842.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 841.45 | 839.91 | 842.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 849.75 | 839.91 | 842.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 851.50 | 842.23 | 843.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 846.25 | 842.23 | 843.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 856.70 | 845.12 | 844.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 858.35 | 847.77 | 845.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 897.15 | 901.15 | 891.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 897.15 | 901.15 | 891.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 897.85 | 900.61 | 893.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 898.90 | 900.61 | 893.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 910.25 | 902.54 | 894.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 917.40 | 905.83 | 897.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 898.50 | 901.87 | 901.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 898.50 | 901.87 | 901.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 09:15:00 | 894.40 | 900.38 | 901.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 888.40 | 887.74 | 892.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 888.40 | 887.74 | 892.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 888.20 | 885.26 | 888.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 888.20 | 885.26 | 888.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 886.00 | 885.41 | 888.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:30:00 | 888.45 | 885.41 | 888.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 11:15:00 | 890.55 | 886.44 | 888.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:00:00 | 890.55 | 886.44 | 888.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 903.60 | 889.87 | 890.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:45:00 | 904.30 | 889.87 | 890.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 901.70 | 892.24 | 891.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-12 14:15:00 | 904.80 | 894.75 | 892.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-13 13:15:00 | 899.75 | 899.92 | 896.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-13 13:30:00 | 901.65 | 899.92 | 896.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 910.25 | 901.98 | 897.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 911.95 | 901.98 | 897.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 918.50 | 934.95 | 937.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 918.50 | 934.95 | 937.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 905.55 | 926.32 | 932.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 14:15:00 | 903.60 | 902.07 | 909.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 15:00:00 | 903.60 | 902.07 | 909.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 909.10 | 903.20 | 908.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:30:00 | 906.20 | 903.20 | 908.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 908.70 | 904.30 | 908.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 909.75 | 904.30 | 908.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 905.60 | 904.56 | 908.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:45:00 | 901.10 | 904.28 | 907.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 09:15:00 | 856.04 | 897.24 | 903.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 14:15:00 | 894.60 | 891.41 | 897.41 | SL hit (close>ema200) qty=0.50 sl=891.41 alert=retest2 |

### Cycle 121 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 900.90 | 897.18 | 896.82 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 895.20 | 897.06 | 897.10 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 901.90 | 896.92 | 896.91 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 895.60 | 896.89 | 897.04 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 911.00 | 896.82 | 896.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 919.00 | 903.20 | 899.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 920.00 | 922.39 | 917.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 920.25 | 922.39 | 917.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 909.60 | 919.83 | 916.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 909.60 | 919.83 | 916.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 906.20 | 917.11 | 915.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 906.20 | 917.11 | 915.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 903.40 | 914.37 | 914.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 10:15:00 | 900.20 | 908.11 | 910.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 15:15:00 | 905.70 | 903.95 | 907.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 09:15:00 | 907.75 | 903.95 | 907.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 926.90 | 908.54 | 909.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 926.90 | 908.54 | 909.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 928.05 | 912.44 | 910.84 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 908.55 | 922.32 | 922.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 10:15:00 | 906.35 | 919.12 | 921.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 919.40 | 918.66 | 920.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 919.40 | 918.66 | 920.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 919.40 | 918.66 | 920.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:15:00 | 920.85 | 918.66 | 920.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 922.55 | 919.44 | 920.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 922.55 | 919.44 | 920.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 925.50 | 920.65 | 921.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 925.50 | 920.65 | 921.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 15:15:00 | 926.25 | 921.77 | 921.59 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 09:15:00 | 919.95 | 921.41 | 921.44 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 923.00 | 921.71 | 921.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 12:15:00 | 927.80 | 922.93 | 922.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 934.50 | 939.13 | 934.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 934.50 | 939.13 | 934.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 934.50 | 939.13 | 934.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:00:00 | 934.50 | 939.13 | 934.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 943.30 | 939.97 | 934.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 14:30:00 | 947.60 | 942.18 | 937.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 927.95 | 936.52 | 936.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 927.95 | 936.52 | 936.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 12:15:00 | 924.50 | 932.18 | 934.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 14:15:00 | 921.00 | 920.88 | 926.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 15:00:00 | 921.00 | 920.88 | 926.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 925.80 | 921.72 | 925.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:00:00 | 925.80 | 921.72 | 925.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 925.85 | 922.55 | 925.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 10:30:00 | 925.25 | 922.55 | 925.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 930.00 | 924.04 | 925.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-24 11:30:00 | 930.80 | 924.04 | 925.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 12:15:00 | 923.90 | 924.01 | 925.80 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 932.00 | 927.21 | 926.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 939.25 | 929.62 | 928.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 949.50 | 955.03 | 949.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 949.50 | 955.03 | 949.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 949.50 | 955.03 | 949.96 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 14:15:00 | 942.55 | 946.87 | 947.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 920.75 | 941.18 | 944.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 940.65 | 932.64 | 937.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 940.65 | 932.64 | 937.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 940.65 | 932.64 | 937.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 938.75 | 932.64 | 937.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 944.65 | 935.04 | 937.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 944.65 | 935.04 | 937.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 941.60 | 936.35 | 938.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 933.00 | 935.36 | 937.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 935.10 | 934.27 | 936.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 935.30 | 935.49 | 936.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:15:00 | 934.70 | 935.49 | 936.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 938.00 | 935.99 | 936.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:00:00 | 938.00 | 935.99 | 936.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 933.90 | 935.58 | 936.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 932.85 | 934.90 | 936.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 886.35 | 919.20 | 928.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 888.35 | 919.20 | 928.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 888.53 | 919.20 | 928.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 887.97 | 919.20 | 928.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 886.21 | 919.20 | 928.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 889.75 | 888.74 | 904.54 | SL hit (close>ema200) qty=0.50 sl=888.74 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 838.75 | 826.38 | 825.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 845.65 | 830.24 | 827.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 818.00 | 831.24 | 829.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 818.00 | 831.24 | 829.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 818.00 | 831.24 | 829.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 816.05 | 831.24 | 829.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 820.75 | 829.14 | 828.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 11:30:00 | 823.35 | 828.07 | 827.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:00:00 | 823.80 | 828.07 | 827.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 12:15:00 | 824.75 | 827.41 | 827.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 824.75 | 827.41 | 827.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 817.30 | 825.39 | 826.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 824.60 | 822.35 | 824.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 824.60 | 822.35 | 824.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 824.60 | 822.35 | 824.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 832.40 | 822.35 | 824.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 831.75 | 824.23 | 825.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 831.75 | 824.23 | 825.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 827.20 | 824.83 | 825.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 820.35 | 824.83 | 825.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 14:15:00 | 779.33 | 794.60 | 807.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 795.00 | 786.40 | 797.34 | SL hit (close>ema200) qty=0.50 sl=786.40 alert=retest2 |

### Cycle 137 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 831.90 | 804.51 | 802.49 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 796.05 | 806.16 | 806.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 769.65 | 792.77 | 799.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 779.25 | 768.69 | 780.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 779.25 | 768.69 | 780.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 779.25 | 768.69 | 780.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 760.20 | 781.25 | 782.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:45:00 | 765.00 | 771.89 | 775.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 787.40 | 777.90 | 776.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 787.40 | 777.90 | 776.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 833.50 | 791.50 | 784.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 823.95 | 824.50 | 811.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 823.95 | 824.50 | 811.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 816.20 | 820.95 | 812.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 815.65 | 820.95 | 812.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 812.50 | 819.26 | 812.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 827.00 | 819.26 | 812.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 811.30 | 824.54 | 820.14 | SL hit (close<static) qty=1.00 sl=812.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 849.80 | 858.47 | 859.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 846.85 | 855.02 | 857.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 880.70 | 857.08 | 857.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 880.70 | 857.08 | 857.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 880.70 | 857.08 | 857.19 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 879.50 | 861.56 | 859.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 890.65 | 870.19 | 863.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 13:15:00 | 886.55 | 887.65 | 878.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 14:00:00 | 886.55 | 887.65 | 878.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 900.55 | 907.93 | 897.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 904.45 | 907.93 | 897.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 912.95 | 918.92 | 913.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:15:00 | 912.35 | 918.92 | 913.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 911.90 | 917.51 | 913.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 929.50 | 912.28 | 911.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-17 09:15:00 | 1397.75 | 2024-05-17 14:15:00 | 1414.55 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-05-17 11:15:00 | 1406.65 | 2024-05-17 14:15:00 | 1414.55 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-05-17 12:00:00 | 1404.05 | 2024-05-17 14:15:00 | 1414.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-05-17 13:00:00 | 1407.90 | 2024-05-17 14:15:00 | 1414.55 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-05-29 11:45:00 | 1453.05 | 2024-05-30 15:15:00 | 1452.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-05-30 15:15:00 | 1452.00 | 2024-05-30 15:15:00 | 1452.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-06-13 11:00:00 | 1483.05 | 2024-06-13 11:15:00 | 1485.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-06-18 09:30:00 | 1507.60 | 2024-06-24 09:15:00 | 1482.85 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-06-18 10:30:00 | 1509.00 | 2024-06-24 09:15:00 | 1482.85 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-06-18 11:00:00 | 1506.15 | 2024-06-24 09:15:00 | 1482.85 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-07-10 10:15:00 | 1424.10 | 2024-07-12 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-10 11:30:00 | 1424.60 | 2024-07-12 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-07-10 15:15:00 | 1425.00 | 2024-07-12 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-07-11 09:30:00 | 1423.40 | 2024-07-12 10:15:00 | 1444.40 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1388.40 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2024-08-06 13:30:00 | 1387.95 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-08-06 14:00:00 | 1388.05 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2024-08-07 09:30:00 | 1383.30 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1344.55 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-08-09 10:45:00 | 1346.90 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-08-09 11:15:00 | 1346.45 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-08-09 12:00:00 | 1346.80 | 2024-08-13 10:15:00 | 1366.75 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-16 10:45:00 | 1340.65 | 2024-08-16 13:15:00 | 1362.45 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-08-23 14:00:00 | 1385.65 | 2024-08-27 12:15:00 | 1379.90 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-26 15:00:00 | 1385.95 | 2024-08-27 12:15:00 | 1379.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-08-27 11:15:00 | 1386.20 | 2024-08-27 12:15:00 | 1379.90 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-09-04 13:15:00 | 1436.40 | 2024-09-05 12:15:00 | 1427.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-09-04 13:45:00 | 1437.10 | 2024-09-05 12:15:00 | 1427.60 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-09-04 15:00:00 | 1436.05 | 2024-09-05 12:15:00 | 1427.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-09-05 09:30:00 | 1436.65 | 2024-09-05 12:15:00 | 1427.60 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-18 11:30:00 | 1478.70 | 2024-09-23 10:15:00 | 1466.50 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-18 13:45:00 | 1483.15 | 2024-09-23 10:15:00 | 1466.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-09-19 09:15:00 | 1492.35 | 2024-09-23 10:15:00 | 1466.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-09-20 09:45:00 | 1478.15 | 2024-09-23 10:15:00 | 1466.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-10-11 11:15:00 | 1343.70 | 2024-10-14 12:15:00 | 1364.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-18 09:15:00 | 1336.90 | 2024-10-18 13:15:00 | 1352.60 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-10-18 10:15:00 | 1343.90 | 2024-10-18 13:15:00 | 1352.60 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-10-18 12:45:00 | 1344.50 | 2024-10-18 13:15:00 | 1352.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1336.20 | 2024-10-23 09:15:00 | 1269.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1336.20 | 2024-10-24 13:15:00 | 1275.35 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2024-10-21 10:30:00 | 1310.00 | 2024-10-25 09:15:00 | 1179.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-06 15:00:00 | 1078.60 | 2024-11-07 09:15:00 | 1064.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1055.80 | 2024-11-18 09:15:00 | 1003.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:45:00 | 1054.95 | 2024-11-18 09:15:00 | 1002.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 10:00:00 | 1054.20 | 2024-11-18 09:15:00 | 1001.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 09:15:00 | 1055.80 | 2024-11-19 09:15:00 | 1022.90 | STOP_HIT | 0.50 | 3.12% |
| SELL | retest2 | 2024-11-12 12:45:00 | 1054.95 | 2024-11-19 09:15:00 | 1022.90 | STOP_HIT | 0.50 | 3.04% |
| SELL | retest2 | 2024-11-13 10:00:00 | 1054.20 | 2024-11-19 09:15:00 | 1022.90 | STOP_HIT | 0.50 | 2.97% |
| SELL | retest2 | 2024-12-26 10:15:00 | 932.30 | 2024-12-27 09:15:00 | 958.00 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-12-26 14:30:00 | 930.30 | 2024-12-27 09:15:00 | 958.00 | STOP_HIT | 1.00 | -2.98% |
| SELL | retest2 | 2025-01-10 09:15:00 | 958.15 | 2025-01-14 13:15:00 | 969.50 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-01-14 10:15:00 | 973.25 | 2025-01-14 13:15:00 | 969.50 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-01-14 10:45:00 | 972.95 | 2025-01-14 13:15:00 | 969.50 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-01-16 09:15:00 | 977.60 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-01-17 09:30:00 | 966.25 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-01-17 10:15:00 | 974.45 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-01-17 11:15:00 | 966.95 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-01-17 12:30:00 | 970.75 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-01-17 14:30:00 | 973.50 | 2025-01-20 09:15:00 | 963.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-01-28 14:45:00 | 937.70 | 2025-01-29 14:15:00 | 947.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-01-29 09:45:00 | 936.00 | 2025-01-29 14:15:00 | 947.60 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-02-10 12:15:00 | 1073.10 | 2025-02-11 10:15:00 | 1056.30 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-02-10 13:00:00 | 1071.35 | 2025-02-11 10:15:00 | 1056.30 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-10 13:30:00 | 1072.10 | 2025-02-11 10:15:00 | 1056.30 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-02-11 09:30:00 | 1072.40 | 2025-02-11 10:15:00 | 1056.30 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-02-13 13:30:00 | 1041.20 | 2025-02-17 15:15:00 | 1050.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1031.80 | 2025-02-17 15:15:00 | 1050.30 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-03-04 12:15:00 | 984.30 | 2025-03-07 12:15:00 | 935.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-04 14:45:00 | 985.50 | 2025-03-07 12:15:00 | 936.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-05 10:00:00 | 984.80 | 2025-03-07 12:15:00 | 935.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-04 12:15:00 | 984.30 | 2025-03-10 09:15:00 | 885.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 14:45:00 | 985.50 | 2025-03-10 09:15:00 | 886.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-05 10:00:00 | 984.80 | 2025-03-10 09:15:00 | 886.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-08 10:30:00 | 676.80 | 2025-04-11 10:15:00 | 682.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-04-08 11:15:00 | 678.65 | 2025-04-11 10:15:00 | 682.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-04-08 13:30:00 | 679.60 | 2025-04-11 10:15:00 | 682.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-04-09 09:15:00 | 678.35 | 2025-04-11 10:15:00 | 682.60 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-04-09 11:00:00 | 672.00 | 2025-04-11 10:15:00 | 682.60 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-04-30 15:00:00 | 837.90 | 2025-05-06 11:15:00 | 829.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-02 09:15:00 | 862.50 | 2025-05-06 11:15:00 | 829.10 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2025-05-08 11:30:00 | 825.60 | 2025-05-13 10:15:00 | 784.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 14:15:00 | 825.00 | 2025-05-13 10:15:00 | 783.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 15:15:00 | 818.60 | 2025-05-13 10:15:00 | 777.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-12 09:30:00 | 819.35 | 2025-05-13 10:15:00 | 778.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 11:30:00 | 825.60 | 2025-05-14 11:15:00 | 782.05 | STOP_HIT | 0.50 | 5.27% |
| SELL | retest2 | 2025-05-08 14:15:00 | 825.00 | 2025-05-14 11:15:00 | 782.05 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-05-08 15:15:00 | 818.60 | 2025-05-14 11:15:00 | 782.05 | STOP_HIT | 0.50 | 4.46% |
| SELL | retest2 | 2025-05-12 09:30:00 | 819.35 | 2025-05-14 11:15:00 | 782.05 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2025-05-16 09:15:00 | 773.55 | 2025-05-16 15:15:00 | 784.70 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-19 09:15:00 | 776.30 | 2025-05-19 09:15:00 | 784.70 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-05-29 09:15:00 | 807.20 | 2025-06-02 15:15:00 | 812.05 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2025-07-02 12:00:00 | 863.15 | 2025-07-02 14:15:00 | 858.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-02 13:00:00 | 865.00 | 2025-07-02 14:15:00 | 858.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-07-09 14:30:00 | 843.60 | 2025-07-11 10:15:00 | 854.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-07-10 10:15:00 | 844.80 | 2025-07-11 10:15:00 | 854.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-10 11:30:00 | 844.30 | 2025-07-11 10:15:00 | 854.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-17 12:15:00 | 871.00 | 2025-07-17 13:15:00 | 866.80 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-21 09:15:00 | 845.00 | 2025-07-28 14:15:00 | 802.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-21 09:15:00 | 845.00 | 2025-07-29 13:15:00 | 806.20 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2025-08-08 09:15:00 | 806.30 | 2025-08-14 13:15:00 | 765.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:15:00 | 806.30 | 2025-08-18 09:15:00 | 790.25 | STOP_HIT | 0.50 | 1.99% |
| SELL | retest2 | 2025-08-26 09:15:00 | 766.05 | 2025-09-02 10:15:00 | 754.90 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-09-03 09:15:00 | 757.00 | 2025-09-05 09:15:00 | 753.05 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-09-09 11:30:00 | 748.15 | 2025-09-19 14:15:00 | 745.15 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-09-10 12:30:00 | 749.05 | 2025-09-19 14:15:00 | 745.15 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-09-11 13:30:00 | 749.05 | 2025-09-19 14:15:00 | 745.15 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-09-11 15:00:00 | 748.20 | 2025-09-19 14:15:00 | 745.15 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-09-12 10:15:00 | 748.20 | 2025-09-19 14:15:00 | 745.15 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-09-26 09:15:00 | 733.80 | 2025-09-30 14:15:00 | 734.70 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-06 14:15:00 | 740.80 | 2025-10-08 15:15:00 | 742.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-10-06 15:15:00 | 741.30 | 2025-10-08 15:15:00 | 742.40 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2025-10-31 11:45:00 | 805.95 | 2025-10-31 15:15:00 | 794.50 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-11 10:15:00 | 824.55 | 2025-11-18 14:15:00 | 847.55 | STOP_HIT | 1.00 | 2.79% |
| BUY | retest2 | 2025-11-11 13:30:00 | 821.35 | 2025-11-18 14:15:00 | 847.55 | STOP_HIT | 1.00 | 3.19% |
| BUY | retest2 | 2025-11-11 14:30:00 | 824.40 | 2025-11-18 14:15:00 | 847.55 | STOP_HIT | 1.00 | 2.81% |
| BUY | retest2 | 2025-11-28 14:15:00 | 857.45 | 2025-12-01 11:15:00 | 848.15 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-12-01 10:00:00 | 856.50 | 2025-12-01 11:15:00 | 848.15 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-05 13:30:00 | 873.30 | 2025-12-08 10:15:00 | 853.65 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-12-10 10:45:00 | 839.25 | 2025-12-12 09:15:00 | 848.10 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-10 13:00:00 | 837.15 | 2025-12-12 09:15:00 | 848.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-10 13:30:00 | 840.00 | 2025-12-12 13:15:00 | 846.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-11 10:30:00 | 835.45 | 2025-12-12 13:15:00 | 846.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-12-11 14:30:00 | 833.20 | 2025-12-12 13:15:00 | 846.30 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-12-11 15:15:00 | 833.85 | 2025-12-12 13:15:00 | 846.30 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-12-18 13:15:00 | 836.90 | 2025-12-19 13:15:00 | 846.35 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-24 10:15:00 | 852.55 | 2025-12-29 11:15:00 | 838.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-26 09:45:00 | 855.60 | 2025-12-29 11:15:00 | 838.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-12-29 09:15:00 | 852.00 | 2025-12-29 11:15:00 | 838.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-06 10:30:00 | 917.40 | 2026-01-07 15:15:00 | 898.50 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2026-01-13 15:15:00 | 911.95 | 2026-01-20 11:15:00 | 918.50 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2026-01-23 12:45:00 | 901.10 | 2026-01-27 09:15:00 | 856.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 12:45:00 | 901.10 | 2026-01-27 14:15:00 | 894.60 | STOP_HIT | 0.50 | 0.72% |
| BUY | retest2 | 2026-02-18 14:30:00 | 947.60 | 2026-02-19 14:15:00 | 927.95 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-03-05 12:45:00 | 933.00 | 2026-03-09 09:15:00 | 886.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:00:00 | 935.10 | 2026-03-09 09:15:00 | 888.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:45:00 | 935.30 | 2026-03-09 09:15:00 | 888.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 11:15:00 | 934.70 | 2026-03-09 09:15:00 | 887.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 932.85 | 2026-03-09 09:15:00 | 886.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 12:45:00 | 933.00 | 2026-03-10 09:15:00 | 889.75 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2026-03-05 15:00:00 | 935.10 | 2026-03-10 09:15:00 | 889.75 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2026-03-06 10:45:00 | 935.30 | 2026-03-10 09:15:00 | 889.75 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2026-03-06 11:15:00 | 934.70 | 2026-03-10 09:15:00 | 889.75 | STOP_HIT | 0.50 | 4.81% |
| SELL | retest2 | 2026-03-06 13:45:00 | 932.85 | 2026-03-10 09:15:00 | 889.75 | STOP_HIT | 0.50 | 4.62% |
| BUY | retest2 | 2026-03-19 11:30:00 | 823.35 | 2026-03-19 12:15:00 | 824.75 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2026-03-19 12:00:00 | 823.80 | 2026-03-19 12:15:00 | 824.75 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-03-20 12:15:00 | 820.35 | 2026-03-23 14:15:00 | 779.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-20 12:15:00 | 820.35 | 2026-03-24 12:15:00 | 795.00 | STOP_HIT | 0.50 | 3.09% |
| SELL | retest2 | 2026-04-02 09:15:00 | 760.20 | 2026-04-06 14:15:00 | 787.40 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2026-04-06 09:45:00 | 765.00 | 2026-04-06 14:15:00 | 787.40 | STOP_HIT | 1.00 | -2.93% |
| BUY | retest2 | 2026-04-10 09:15:00 | 827.00 | 2026-04-13 09:15:00 | 811.30 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-13 11:45:00 | 816.25 | 2026-04-24 10:15:00 | 849.80 | STOP_HIT | 1.00 | 4.11% |
| BUY | retest2 | 2026-04-13 13:00:00 | 816.50 | 2026-04-24 10:15:00 | 849.80 | STOP_HIT | 1.00 | 4.08% |
| BUY | retest2 | 2026-04-13 13:30:00 | 817.95 | 2026-04-24 10:15:00 | 849.80 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2026-04-13 15:15:00 | 821.90 | 2026-04-24 10:15:00 | 849.80 | STOP_HIT | 1.00 | 3.39% |
