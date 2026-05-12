# Info Edge (India) Ltd. (NAUKRI)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 978.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 73 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 25 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 72 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 20 / 59
- **Target hits / Stop hits / Partials:** 0 / 71 / 8
- **Avg / median % per leg:** -0.07% / -0.75%
- **Sum % (uncompounded):** -5.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.32% | -31.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.32% | -31.6% |
| SELL (all) | 55 | 20 | 36.4% | 0 | 47 | 8 | 0.48% | 26.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 55 | 20 | 36.4% | 0 | 47 | 8 | 0.48% | 26.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 79 | 20 | 25.3% | 0 | 71 | 8 | -0.07% | -5.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1440.50 | 1388.35 | 1386.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 1445.50 | 1399.78 | 1391.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1460.00 | 1462.88 | 1438.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 1460.00 | 1462.88 | 1438.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1453.00 | 1462.26 | 1445.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 1476.00 | 1459.53 | 1451.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:00:00 | 1483.50 | 1464.33 | 1454.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:15:00 | 1487.50 | 1469.31 | 1458.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 1475.50 | 1496.63 | 1485.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1473.50 | 1492.01 | 1484.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 1476.50 | 1492.01 | 1484.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1470.50 | 1482.22 | 1481.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1470.50 | 1482.22 | 1481.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 1468.00 | 1479.37 | 1480.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 1468.00 | 1479.37 | 1480.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 1457.50 | 1471.37 | 1476.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1467.50 | 1456.77 | 1464.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 1467.50 | 1456.77 | 1464.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1467.50 | 1456.77 | 1464.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1467.50 | 1456.77 | 1464.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1453.00 | 1456.02 | 1463.24 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 1471.00 | 1465.78 | 1465.71 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 1464.00 | 1465.42 | 1465.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 11:15:00 | 1459.00 | 1464.14 | 1464.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 13:15:00 | 1471.00 | 1464.29 | 1464.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 13:15:00 | 1471.00 | 1464.29 | 1464.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1471.00 | 1464.29 | 1464.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 1466.50 | 1464.29 | 1464.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1460.50 | 1463.53 | 1464.42 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1483.00 | 1467.26 | 1465.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1490.00 | 1473.77 | 1469.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1482.00 | 1482.93 | 1476.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 10:15:00 | 1482.00 | 1482.93 | 1476.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1482.00 | 1482.93 | 1476.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 1482.00 | 1482.93 | 1476.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1472.50 | 1480.84 | 1476.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:00:00 | 1472.50 | 1480.84 | 1476.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1473.50 | 1479.37 | 1476.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:00:00 | 1473.50 | 1479.37 | 1476.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1473.00 | 1476.22 | 1475.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1469.50 | 1476.22 | 1475.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1479.50 | 1477.80 | 1476.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:30:00 | 1474.50 | 1477.80 | 1476.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1478.00 | 1477.84 | 1476.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 12:00:00 | 1478.00 | 1477.84 | 1476.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1470.50 | 1476.37 | 1475.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 1470.50 | 1476.37 | 1475.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 1455.50 | 1472.20 | 1474.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 1416.50 | 1455.78 | 1465.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 1437.00 | 1436.86 | 1448.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 1437.00 | 1436.86 | 1448.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1437.00 | 1436.86 | 1448.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 1406.00 | 1428.63 | 1436.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:30:00 | 1416.90 | 1421.47 | 1430.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 1415.70 | 1424.77 | 1429.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:30:00 | 1409.50 | 1421.39 | 1425.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1425.20 | 1415.55 | 1419.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 1425.20 | 1415.55 | 1419.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1433.20 | 1419.08 | 1420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:00:00 | 1433.20 | 1419.08 | 1420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 1469.00 | 1429.06 | 1424.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1469.00 | 1429.06 | 1424.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 1475.90 | 1438.43 | 1429.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 1525.00 | 1530.21 | 1516.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:30:00 | 1521.40 | 1530.21 | 1516.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1509.10 | 1524.62 | 1518.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1509.10 | 1524.62 | 1518.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1505.80 | 1520.85 | 1516.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:30:00 | 1507.00 | 1520.85 | 1516.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1497.90 | 1514.21 | 1514.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 1494.60 | 1505.62 | 1510.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 1477.40 | 1465.58 | 1478.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1477.40 | 1465.58 | 1478.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1477.40 | 1465.58 | 1478.06 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1494.40 | 1484.51 | 1483.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1477.80 | 1482.53 | 1482.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1465.90 | 1476.47 | 1479.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 10:15:00 | 1471.80 | 1471.40 | 1475.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-19 11:15:00 | 1474.00 | 1471.40 | 1475.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1480.00 | 1473.12 | 1475.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1480.00 | 1473.12 | 1475.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1471.30 | 1472.76 | 1475.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 1476.80 | 1472.76 | 1475.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1469.60 | 1472.31 | 1474.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 15:15:00 | 1461.00 | 1472.31 | 1474.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1489.40 | 1473.92 | 1474.99 | SL hit (close>static) qty=1.00 sl=1478.10 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1492.40 | 1477.61 | 1476.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 11:15:00 | 1502.60 | 1482.61 | 1478.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1478.80 | 1492.18 | 1486.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1478.80 | 1492.18 | 1486.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1478.80 | 1492.18 | 1486.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1478.80 | 1492.18 | 1486.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1487.20 | 1491.19 | 1486.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:30:00 | 1483.20 | 1491.19 | 1486.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1490.90 | 1491.13 | 1486.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1509.90 | 1487.37 | 1486.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 1505.50 | 1509.61 | 1507.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 10:15:00 | 1483.40 | 1508.06 | 1510.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 10:15:00 | 1483.40 | 1508.06 | 1510.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 13:15:00 | 1481.10 | 1496.19 | 1503.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 13:15:00 | 1460.60 | 1451.84 | 1466.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 14:00:00 | 1460.60 | 1451.84 | 1466.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1460.40 | 1454.00 | 1464.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:15:00 | 1441.40 | 1453.48 | 1460.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 1469.30 | 1454.71 | 1459.45 | SL hit (close>static) qty=1.00 sl=1467.80 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 1475.60 | 1462.68 | 1462.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 1485.00 | 1470.03 | 1466.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 1468.40 | 1474.17 | 1469.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1468.40 | 1474.17 | 1469.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1468.40 | 1474.17 | 1469.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1468.40 | 1474.17 | 1469.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 10:15:00 | 1405.80 | 1460.50 | 1463.57 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1459.40 | 1455.24 | 1454.69 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 1443.10 | 1452.85 | 1453.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 1432.00 | 1446.02 | 1449.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 1396.80 | 1381.50 | 1397.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 13:15:00 | 1396.80 | 1381.50 | 1397.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 13:15:00 | 1396.80 | 1381.50 | 1397.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:00:00 | 1396.80 | 1381.50 | 1397.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1397.80 | 1384.76 | 1397.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1397.80 | 1384.76 | 1397.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1398.00 | 1387.41 | 1397.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:15:00 | 1392.40 | 1387.41 | 1397.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 13:15:00 | 1400.70 | 1392.16 | 1395.85 | SL hit (close>static) qty=1.00 sl=1400.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 1398.70 | 1382.24 | 1381.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 15:15:00 | 1405.80 | 1389.47 | 1384.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 1463.70 | 1466.67 | 1447.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 1463.70 | 1466.67 | 1447.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1459.00 | 1461.93 | 1452.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:45:00 | 1456.20 | 1461.93 | 1452.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1451.30 | 1459.80 | 1452.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1441.80 | 1459.80 | 1452.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1430.70 | 1453.98 | 1450.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 1430.70 | 1453.98 | 1450.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 11:15:00 | 1432.00 | 1445.68 | 1446.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 1427.10 | 1437.13 | 1441.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 1400.40 | 1400.30 | 1413.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:45:00 | 1400.00 | 1400.30 | 1413.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1409.10 | 1402.06 | 1413.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 1408.40 | 1402.06 | 1413.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1399.10 | 1392.72 | 1401.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:45:00 | 1402.10 | 1392.72 | 1401.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1373.40 | 1388.42 | 1398.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 1368.00 | 1380.83 | 1388.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-04 13:00:00 | 1369.20 | 1371.25 | 1379.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:15:00 | 1365.50 | 1371.84 | 1378.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 15:15:00 | 1370.00 | 1348.26 | 1346.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1370.00 | 1348.26 | 1346.46 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 1331.90 | 1345.26 | 1345.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 10:15:00 | 1324.50 | 1337.46 | 1341.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 09:15:00 | 1338.10 | 1327.02 | 1332.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 09:15:00 | 1338.10 | 1327.02 | 1332.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 1338.10 | 1327.02 | 1332.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 1338.10 | 1327.02 | 1332.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1327.30 | 1327.08 | 1332.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:30:00 | 1321.40 | 1324.78 | 1330.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1347.60 | 1326.97 | 1326.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 1347.60 | 1326.97 | 1326.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 10:15:00 | 1368.70 | 1335.32 | 1330.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 1356.90 | 1362.57 | 1350.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 1356.90 | 1362.57 | 1350.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1351.80 | 1360.06 | 1352.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 1351.80 | 1360.06 | 1352.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 1355.50 | 1359.15 | 1352.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 1350.10 | 1359.15 | 1352.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1355.10 | 1358.34 | 1352.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1345.00 | 1358.34 | 1352.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1359.50 | 1358.57 | 1353.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 1356.10 | 1358.57 | 1353.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1385.10 | 1404.82 | 1395.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 1385.10 | 1404.82 | 1395.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1382.10 | 1400.27 | 1394.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:00:00 | 1382.10 | 1400.27 | 1394.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 1380.80 | 1390.76 | 1390.96 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 1415.70 | 1394.35 | 1392.47 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 1386.70 | 1396.04 | 1396.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1383.10 | 1389.82 | 1393.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1390.00 | 1387.53 | 1391.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1390.00 | 1387.53 | 1391.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1372.70 | 1384.94 | 1389.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 1368.60 | 1384.94 | 1389.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 1370.50 | 1370.40 | 1376.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 1367.50 | 1347.65 | 1348.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1364.30 | 1350.98 | 1349.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1364.30 | 1350.98 | 1349.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1388.20 | 1358.28 | 1353.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1360.10 | 1374.26 | 1366.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1360.10 | 1374.26 | 1366.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1360.10 | 1374.26 | 1366.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 1360.10 | 1374.26 | 1366.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1367.20 | 1372.84 | 1366.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 1367.80 | 1372.82 | 1367.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:00:00 | 1372.70 | 1372.82 | 1367.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 13:45:00 | 1369.50 | 1372.01 | 1367.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 1367.60 | 1370.82 | 1368.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1365.00 | 1369.66 | 1368.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1359.20 | 1367.57 | 1367.53 | SL hit (close<static) qty=1.00 sl=1359.70 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 1360.00 | 1366.05 | 1366.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 1352.70 | 1361.23 | 1364.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1359.50 | 1359.36 | 1362.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 1359.50 | 1359.36 | 1362.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1357.00 | 1358.89 | 1362.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1359.20 | 1358.89 | 1362.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1362.10 | 1359.53 | 1362.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1362.10 | 1359.53 | 1362.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1365.00 | 1360.63 | 1362.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1355.00 | 1360.63 | 1362.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1363.60 | 1361.22 | 1362.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 12:45:00 | 1350.80 | 1358.69 | 1361.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 1349.20 | 1355.54 | 1359.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 1396.10 | 1363.73 | 1362.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1396.10 | 1363.73 | 1362.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 11:15:00 | 1414.40 | 1390.14 | 1382.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 1401.10 | 1404.83 | 1393.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:45:00 | 1398.10 | 1404.83 | 1393.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1398.40 | 1402.93 | 1394.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 1390.60 | 1402.93 | 1394.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1393.30 | 1400.33 | 1395.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1392.10 | 1400.33 | 1395.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 1394.40 | 1399.14 | 1395.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 1387.40 | 1399.14 | 1395.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1377.50 | 1394.81 | 1393.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1377.50 | 1394.81 | 1393.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1380.20 | 1391.89 | 1392.60 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1396.30 | 1392.74 | 1392.73 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1386.00 | 1391.39 | 1392.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1358.20 | 1383.73 | 1388.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 13:15:00 | 1315.70 | 1314.79 | 1326.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 14:00:00 | 1315.70 | 1314.79 | 1326.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1310.90 | 1313.09 | 1322.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1315.30 | 1313.09 | 1322.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1336.00 | 1317.67 | 1323.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 1336.00 | 1317.67 | 1323.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1334.80 | 1321.10 | 1324.71 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1338.80 | 1329.15 | 1327.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 1351.80 | 1337.00 | 1333.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1372.40 | 1378.15 | 1366.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 1372.40 | 1378.15 | 1366.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1365.40 | 1375.60 | 1366.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1365.40 | 1375.60 | 1366.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1370.20 | 1374.52 | 1366.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 1366.20 | 1374.52 | 1366.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1369.90 | 1373.60 | 1366.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 1368.20 | 1373.60 | 1366.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1372.20 | 1373.32 | 1367.42 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 10:15:00 | 1350.00 | 1363.45 | 1364.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 1336.00 | 1344.77 | 1351.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 14:15:00 | 1334.30 | 1332.91 | 1339.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 14:30:00 | 1331.90 | 1332.91 | 1339.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1342.50 | 1335.00 | 1339.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:15:00 | 1342.30 | 1335.00 | 1339.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1339.00 | 1335.80 | 1339.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:30:00 | 1336.70 | 1333.28 | 1338.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 11:00:00 | 1336.20 | 1330.13 | 1333.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:45:00 | 1336.70 | 1332.61 | 1333.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 09:45:00 | 1336.30 | 1333.85 | 1334.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 10:15:00 | 1345.50 | 1336.18 | 1335.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1345.50 | 1336.18 | 1335.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1350.80 | 1339.11 | 1336.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 1336.50 | 1344.41 | 1341.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 1336.50 | 1344.41 | 1341.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1336.50 | 1344.41 | 1341.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1335.20 | 1344.41 | 1341.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1340.00 | 1343.53 | 1340.96 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1325.00 | 1338.66 | 1339.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 1314.20 | 1328.67 | 1333.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1320.50 | 1319.79 | 1326.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-21 13:45:00 | 1316.40 | 1319.79 | 1326.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1344.20 | 1324.05 | 1327.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 1345.00 | 1324.05 | 1327.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1353.50 | 1329.94 | 1329.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 1368.60 | 1337.67 | 1333.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 14:15:00 | 1377.00 | 1381.88 | 1366.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 14:45:00 | 1377.00 | 1381.88 | 1366.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1372.90 | 1379.62 | 1368.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 1372.90 | 1379.62 | 1368.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1365.80 | 1376.86 | 1367.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 1365.80 | 1376.86 | 1367.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 1368.00 | 1375.09 | 1367.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 1371.30 | 1373.49 | 1367.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 1361.00 | 1370.99 | 1367.28 | SL hit (close<static) qty=1.00 sl=1363.50 alert=retest2 |

### Cycle 36 — SELL (started 2025-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 15:15:00 | 1378.80 | 1382.23 | 1382.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1345.30 | 1374.84 | 1378.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1336.00 | 1333.58 | 1344.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 1336.00 | 1333.58 | 1344.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1334.40 | 1330.43 | 1336.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1338.80 | 1330.43 | 1336.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1340.70 | 1332.49 | 1336.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 1341.30 | 1332.49 | 1336.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1344.50 | 1334.89 | 1337.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1340.00 | 1334.89 | 1337.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:30:00 | 1334.10 | 1337.21 | 1337.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 1342.60 | 1338.18 | 1338.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 1342.60 | 1338.18 | 1338.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1348.00 | 1340.14 | 1339.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1340.00 | 1340.11 | 1339.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1340.00 | 1340.11 | 1339.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1340.00 | 1340.11 | 1339.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:30:00 | 1342.50 | 1340.11 | 1339.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1334.50 | 1338.99 | 1338.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1335.60 | 1338.99 | 1338.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 1331.70 | 1337.53 | 1338.08 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1340.90 | 1338.26 | 1338.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 1344.00 | 1339.41 | 1338.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1343.20 | 1356.36 | 1350.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1343.20 | 1356.36 | 1350.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1343.20 | 1356.36 | 1350.06 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1338.00 | 1349.58 | 1349.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 1328.00 | 1343.70 | 1346.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 1332.40 | 1328.74 | 1334.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 13:15:00 | 1332.40 | 1328.74 | 1334.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 1332.40 | 1328.74 | 1334.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 1332.40 | 1328.74 | 1334.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1331.80 | 1329.35 | 1334.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1332.00 | 1329.35 | 1334.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1334.10 | 1330.30 | 1334.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1322.30 | 1330.30 | 1334.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1337.50 | 1327.94 | 1330.02 | SL hit (close>static) qty=1.00 sl=1334.80 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 1351.50 | 1334.90 | 1332.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 12:15:00 | 1360.80 | 1340.08 | 1335.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1350.60 | 1359.31 | 1352.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1350.60 | 1359.31 | 1352.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1350.60 | 1359.31 | 1352.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 1346.80 | 1359.31 | 1352.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1350.00 | 1357.45 | 1352.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 1348.80 | 1357.45 | 1352.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1347.60 | 1354.51 | 1351.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:00:00 | 1347.60 | 1354.51 | 1351.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1345.60 | 1352.73 | 1351.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:45:00 | 1344.20 | 1352.73 | 1351.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 1335.00 | 1347.43 | 1349.11 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 12:15:00 | 1356.70 | 1349.82 | 1349.74 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 15:15:00 | 1338.00 | 1348.32 | 1349.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 1331.20 | 1338.90 | 1343.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1344.10 | 1339.31 | 1342.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1344.10 | 1339.31 | 1342.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1344.10 | 1339.31 | 1342.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1320.20 | 1339.03 | 1341.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:30:00 | 1325.10 | 1331.83 | 1336.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:00:00 | 1325.10 | 1330.49 | 1335.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 12:30:00 | 1325.20 | 1329.57 | 1334.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1336.40 | 1331.35 | 1333.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1348.40 | 1335.79 | 1335.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1348.40 | 1335.79 | 1335.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1354.20 | 1341.35 | 1338.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1368.90 | 1376.06 | 1363.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 1368.90 | 1376.06 | 1363.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1371.30 | 1375.11 | 1364.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:15:00 | 1362.40 | 1375.11 | 1364.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1376.00 | 1375.29 | 1365.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 1377.00 | 1375.43 | 1367.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 1377.00 | 1375.42 | 1368.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 10:15:00 | 1377.90 | 1386.65 | 1384.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 1363.70 | 1380.11 | 1381.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1363.70 | 1380.11 | 1381.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1356.20 | 1370.53 | 1376.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1371.30 | 1368.94 | 1374.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1371.30 | 1368.94 | 1374.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1370.20 | 1369.19 | 1373.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 1374.80 | 1369.19 | 1373.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1374.80 | 1370.31 | 1373.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 1375.30 | 1370.31 | 1373.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 1377.00 | 1371.65 | 1374.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 1377.00 | 1371.65 | 1374.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 1370.10 | 1371.34 | 1373.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1360.10 | 1370.16 | 1372.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1375.60 | 1359.39 | 1357.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1375.60 | 1359.39 | 1357.95 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 12:15:00 | 1355.10 | 1358.55 | 1358.79 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 13:15:00 | 1360.90 | 1359.02 | 1358.98 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 10:15:00 | 1343.20 | 1356.24 | 1357.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1340.80 | 1347.56 | 1351.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1343.50 | 1341.45 | 1346.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:45:00 | 1344.00 | 1341.45 | 1346.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1345.50 | 1342.26 | 1346.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1345.50 | 1342.26 | 1346.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1338.70 | 1341.55 | 1345.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1333.10 | 1340.04 | 1344.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 1352.90 | 1338.81 | 1339.64 | SL hit (close>static) qty=1.00 sl=1346.20 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1349.60 | 1340.97 | 1340.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 1360.00 | 1347.68 | 1344.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 1366.80 | 1368.81 | 1361.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:45:00 | 1366.00 | 1368.81 | 1361.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1364.80 | 1367.33 | 1362.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1364.80 | 1367.33 | 1362.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1366.60 | 1367.19 | 1362.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:15:00 | 1369.30 | 1367.19 | 1362.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 1368.40 | 1367.14 | 1363.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 1373.00 | 1366.47 | 1363.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 1357.80 | 1364.15 | 1363.46 | SL hit (close<static) qty=1.00 sl=1361.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1353.80 | 1362.08 | 1362.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1344.30 | 1356.87 | 1359.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 10:15:00 | 1339.30 | 1336.94 | 1342.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:00:00 | 1339.30 | 1336.94 | 1342.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1340.10 | 1338.05 | 1341.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1340.10 | 1338.05 | 1341.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1340.50 | 1338.54 | 1341.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 1350.40 | 1341.09 | 1341.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1362.00 | 1345.28 | 1343.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 12:15:00 | 1371.00 | 1352.95 | 1347.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1345.30 | 1356.84 | 1351.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1345.30 | 1356.84 | 1351.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1345.30 | 1356.84 | 1351.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1345.30 | 1356.84 | 1351.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1343.50 | 1354.17 | 1351.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 1343.50 | 1354.17 | 1351.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1342.10 | 1349.70 | 1349.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 1342.10 | 1349.70 | 1349.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 1341.00 | 1347.96 | 1348.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1329.80 | 1343.13 | 1346.02 | Break + close below crossover candle low |

### Cycle 55 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 1374.30 | 1347.25 | 1346.97 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1342.10 | 1350.49 | 1351.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1338.90 | 1348.18 | 1350.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1318.50 | 1316.05 | 1324.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1318.50 | 1316.05 | 1324.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1318.50 | 1316.05 | 1324.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 1302.30 | 1317.49 | 1320.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 1305.80 | 1315.16 | 1319.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 1332.00 | 1321.48 | 1320.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 14:15:00 | 1332.00 | 1321.48 | 1320.52 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1314.00 | 1319.59 | 1320.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1308.40 | 1316.06 | 1318.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1339.60 | 1308.25 | 1310.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1339.60 | 1308.25 | 1310.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1339.60 | 1308.25 | 1310.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1339.60 | 1308.25 | 1310.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1324.70 | 1311.54 | 1311.92 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 1333.40 | 1315.91 | 1313.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 1337.40 | 1322.78 | 1317.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 1319.20 | 1322.07 | 1317.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 10:15:00 | 1319.20 | 1322.07 | 1317.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1319.20 | 1322.07 | 1317.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1319.20 | 1322.07 | 1317.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 1307.40 | 1319.13 | 1316.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 1301.10 | 1319.13 | 1316.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1302.60 | 1315.83 | 1315.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 1306.00 | 1315.83 | 1315.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1321.10 | 1317.33 | 1316.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 1328.80 | 1319.82 | 1317.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1309.80 | 1317.82 | 1316.80 | SL hit (close<static) qty=1.00 sl=1315.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 1306.90 | 1315.63 | 1315.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1300.00 | 1312.51 | 1314.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 11:15:00 | 1282.80 | 1279.77 | 1290.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 12:00:00 | 1282.80 | 1279.77 | 1290.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1298.60 | 1282.85 | 1289.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 14:45:00 | 1298.90 | 1282.85 | 1289.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1301.30 | 1286.54 | 1290.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1283.50 | 1286.54 | 1290.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 10:15:00 | 1219.33 | 1247.87 | 1257.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 1237.50 | 1237.07 | 1248.19 | SL hit (close>ema200) qty=0.50 sl=1237.07 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 1195.70 | 1171.79 | 1169.17 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1169.80 | 1176.65 | 1177.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1140.30 | 1169.38 | 1173.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 13:15:00 | 1136.00 | 1134.82 | 1147.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 14:00:00 | 1136.00 | 1134.82 | 1147.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 1135.00 | 1135.25 | 1145.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1125.00 | 1135.25 | 1145.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:45:00 | 1123.00 | 1135.10 | 1143.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 12:15:00 | 1127.80 | 1134.70 | 1142.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1128.00 | 1130.38 | 1136.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1124.00 | 1129.22 | 1134.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1132.50 | 1129.22 | 1134.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1103.30 | 1123.42 | 1130.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:15:00 | 1102.50 | 1123.42 | 1130.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:45:00 | 1098.70 | 1114.97 | 1124.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:45:00 | 1101.40 | 1110.94 | 1117.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 14:15:00 | 1071.41 | 1087.64 | 1100.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 14:15:00 | 1071.60 | 1087.64 | 1100.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 1068.75 | 1085.01 | 1097.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 15:15:00 | 1066.85 | 1085.01 | 1097.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 1072.00 | 1069.72 | 1082.12 | SL hit (close>ema200) qty=0.50 sl=1069.72 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 971.70 | 955.63 | 953.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 995.60 | 965.34 | 958.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 973.30 | 984.89 | 974.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 973.30 | 984.89 | 974.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 973.30 | 984.89 | 974.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 972.00 | 984.89 | 974.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 969.80 | 981.87 | 974.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:45:00 | 970.10 | 981.87 | 974.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 978.00 | 981.10 | 974.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:15:00 | 980.50 | 981.10 | 974.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 989.40 | 975.70 | 973.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 15:15:00 | 988.90 | 984.36 | 980.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 958.40 | 979.89 | 978.85 | SL hit (close<static) qty=1.00 sl=968.60 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 970.00 | 977.92 | 978.04 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 992.00 | 978.16 | 976.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1015.60 | 990.52 | 983.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1000.50 | 1003.54 | 994.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:30:00 | 1001.30 | 1003.54 | 994.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 997.00 | 1002.24 | 994.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1005.30 | 1002.24 | 994.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 996.90 | 1001.17 | 994.56 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 959.30 | 990.03 | 992.56 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 997.50 | 982.68 | 981.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 1003.45 | 993.67 | 989.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 991.10 | 1016.38 | 1008.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 09:15:00 | 991.10 | 1016.38 | 1008.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 991.10 | 1016.38 | 1008.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 991.10 | 1016.38 | 1008.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 10:15:00 | 997.00 | 1012.51 | 1007.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 1004.05 | 1011.65 | 1007.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 14:45:00 | 1002.90 | 1007.81 | 1006.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 978.25 | 1001.13 | 1003.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 09:15:00 | 978.25 | 1001.13 | 1003.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 10:15:00 | 971.20 | 995.14 | 1000.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 14:15:00 | 990.65 | 988.55 | 995.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 15:00:00 | 990.65 | 988.55 | 995.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 996.60 | 989.93 | 994.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 996.60 | 989.93 | 994.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 1000.00 | 991.94 | 994.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:00:00 | 1000.00 | 991.94 | 994.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 1001.40 | 993.83 | 995.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 996.30 | 995.24 | 995.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1011.20 | 997.59 | 996.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1011.20 | 997.59 | 996.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1038.00 | 1021.25 | 1010.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 1054.60 | 1069.31 | 1062.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 09:15:00 | 1054.60 | 1069.31 | 1062.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1054.60 | 1069.31 | 1062.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 1054.60 | 1069.31 | 1062.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 1059.75 | 1067.40 | 1062.26 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 13:15:00 | 1046.80 | 1057.71 | 1058.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1028.00 | 1049.41 | 1054.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1012.70 | 999.40 | 1014.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1012.70 | 999.40 | 1014.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1012.70 | 999.40 | 1014.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1017.00 | 999.40 | 1014.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1012.25 | 1001.97 | 1014.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1018.80 | 1001.97 | 1014.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1008.05 | 1003.99 | 1009.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 1010.30 | 1003.99 | 1009.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1009.95 | 1005.18 | 1009.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:45:00 | 1009.15 | 1005.18 | 1009.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1006.30 | 1005.40 | 1009.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:45:00 | 1004.35 | 1004.97 | 1008.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 15:00:00 | 1003.35 | 1004.65 | 1008.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 09:15:00 | 1013.45 | 1007.25 | 1008.85 | SL hit (close>static) qty=1.00 sl=1010.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 15:15:00 | 988.95 | 976.79 | 976.55 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 970.00 | 975.43 | 975.96 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 980.80 | 976.51 | 976.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 12:15:00 | 988.00 | 979.50 | 977.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 975.00 | 980.48 | 979.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 975.00 | 980.48 | 979.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 975.00 | 980.48 | 979.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 975.50 | 980.48 | 979.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 982.00 | 980.79 | 979.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 984.20 | 981.04 | 979.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:30:00 | 1476.00 | 2025-05-19 14:15:00 | 1468.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-05-15 12:00:00 | 1483.50 | 2025-05-19 14:15:00 | 1468.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-15 14:15:00 | 1487.50 | 2025-05-19 14:15:00 | 1468.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-05-19 10:15:00 | 1475.50 | 2025-05-19 14:15:00 | 1468.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-06-02 09:15:00 | 1406.00 | 2025-06-05 11:15:00 | 1469.00 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-06-02 11:30:00 | 1416.90 | 2025-06-05 11:15:00 | 1469.00 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2025-06-03 09:45:00 | 1415.70 | 2025-06-05 11:15:00 | 1469.00 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2025-06-04 09:30:00 | 1409.50 | 2025-06-05 11:15:00 | 1469.00 | STOP_HIT | 1.00 | -4.22% |
| SELL | retest2 | 2025-06-19 15:15:00 | 1461.00 | 2025-06-20 09:15:00 | 1489.40 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1509.90 | 2025-06-30 10:15:00 | 1483.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-06-26 13:00:00 | 1505.50 | 2025-06-30 10:15:00 | 1483.40 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-07-03 15:15:00 | 1441.40 | 2025-07-04 09:15:00 | 1469.30 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-07-15 09:15:00 | 1392.40 | 2025-07-15 13:15:00 | 1400.70 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-16 09:30:00 | 1393.90 | 2025-07-21 13:15:00 | 1398.70 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-16 10:45:00 | 1394.00 | 2025-07-21 13:15:00 | 1398.70 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-07-16 15:00:00 | 1392.00 | 2025-07-21 13:15:00 | 1398.70 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-07-17 10:30:00 | 1392.10 | 2025-07-21 13:15:00 | 1398.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-08-01 14:30:00 | 1368.00 | 2025-08-07 15:15:00 | 1370.00 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-08-04 13:00:00 | 1369.20 | 2025-08-07 15:15:00 | 1370.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-08-05 09:15:00 | 1365.50 | 2025-08-07 15:15:00 | 1370.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-08-12 11:30:00 | 1321.40 | 2025-08-14 09:15:00 | 1347.60 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-29 13:15:00 | 1368.60 | 2025-09-09 10:15:00 | 1364.30 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-09-01 15:00:00 | 1370.50 | 2025-09-09 10:15:00 | 1364.30 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2025-09-09 10:00:00 | 1367.50 | 2025-09-09 10:15:00 | 1364.30 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-09-11 11:30:00 | 1367.80 | 2025-09-12 12:15:00 | 1359.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-09-11 12:00:00 | 1372.70 | 2025-09-12 12:15:00 | 1359.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-11 13:45:00 | 1369.50 | 2025-09-12 12:15:00 | 1359.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-12 11:15:00 | 1367.60 | 2025-09-12 12:15:00 | 1359.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-16 12:45:00 | 1350.80 | 2025-09-17 09:15:00 | 1396.10 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-09-16 14:45:00 | 1349.20 | 2025-09-17 09:15:00 | 1396.10 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2025-10-14 11:30:00 | 1336.70 | 2025-10-16 10:15:00 | 1345.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-15 11:00:00 | 1336.20 | 2025-10-16 10:15:00 | 1345.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-10-15 14:45:00 | 1336.70 | 2025-10-16 10:15:00 | 1345.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-16 09:45:00 | 1336.30 | 2025-10-16 10:15:00 | 1345.50 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-10-27 12:45:00 | 1371.30 | 2025-10-27 13:15:00 | 1361.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1388.20 | 2025-10-31 15:15:00 | 1378.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1340.00 | 2025-11-10 14:15:00 | 1342.60 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-11-10 11:30:00 | 1334.10 | 2025-11-10 14:15:00 | 1342.60 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1322.30 | 2025-11-19 09:15:00 | 1337.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1320.20 | 2025-12-01 11:15:00 | 1348.40 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-11-28 10:30:00 | 1325.10 | 2025-12-01 11:15:00 | 1348.40 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-28 12:00:00 | 1325.10 | 2025-12-01 11:15:00 | 1348.40 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-28 12:30:00 | 1325.20 | 2025-12-01 11:15:00 | 1348.40 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-12-03 14:15:00 | 1377.00 | 2025-12-08 11:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-03 15:15:00 | 1377.00 | 2025-12-08 11:15:00 | 1363.70 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-08 10:15:00 | 1377.90 | 2025-12-08 11:15:00 | 1363.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1360.10 | 2025-12-12 13:15:00 | 1375.60 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-18 14:30:00 | 1333.10 | 2025-12-22 09:15:00 | 1352.90 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-26 11:15:00 | 1369.30 | 2025-12-29 11:15:00 | 1357.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-12-26 13:15:00 | 1368.40 | 2025-12-29 11:15:00 | 1357.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1373.00 | 2025-12-29 11:15:00 | 1357.80 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2026-01-14 14:15:00 | 1302.30 | 2026-01-16 14:15:00 | 1332.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-14 15:00:00 | 1305.80 | 2026-01-16 14:15:00 | 1332.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-01-23 10:45:00 | 1328.80 | 2026-01-23 11:15:00 | 1309.80 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1283.50 | 2026-02-02 10:15:00 | 1219.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1283.50 | 2026-02-02 14:15:00 | 1237.50 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1125.00 | 2026-02-20 14:15:00 | 1071.41 | PARTIAL | 0.50 | 4.76% |
| SELL | retest2 | 2026-02-16 10:45:00 | 1123.00 | 2026-02-20 14:15:00 | 1071.60 | PARTIAL | 0.50 | 4.58% |
| SELL | retest2 | 2026-02-16 12:15:00 | 1127.80 | 2026-02-20 15:15:00 | 1068.75 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-02-17 11:00:00 | 1128.00 | 2026-02-20 15:15:00 | 1066.85 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1125.00 | 2026-02-23 15:15:00 | 1072.00 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2026-02-16 10:45:00 | 1123.00 | 2026-02-23 15:15:00 | 1072.00 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2026-02-16 12:15:00 | 1127.80 | 2026-02-23 15:15:00 | 1072.00 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2026-02-17 11:00:00 | 1128.00 | 2026-02-23 15:15:00 | 1072.00 | STOP_HIT | 0.50 | 4.96% |
| SELL | retest2 | 2026-02-18 10:15:00 | 1102.50 | 2026-02-24 09:15:00 | 1047.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 12:45:00 | 1098.70 | 2026-02-24 09:15:00 | 1043.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1101.40 | 2026-02-24 09:15:00 | 1046.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-18 10:15:00 | 1102.50 | 2026-02-25 09:15:00 | 1044.30 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2026-02-18 12:45:00 | 1098.70 | 2026-02-25 09:15:00 | 1044.30 | STOP_HIT | 0.50 | 4.95% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1101.40 | 2026-02-25 09:15:00 | 1044.30 | STOP_HIT | 0.50 | 5.18% |
| BUY | retest2 | 2026-03-19 12:15:00 | 980.50 | 2026-03-23 09:15:00 | 958.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-03-20 09:15:00 | 989.40 | 2026-03-23 09:15:00 | 958.40 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-03-20 15:15:00 | 988.90 | 2026-03-23 09:15:00 | 958.40 | STOP_HIT | 1.00 | -3.08% |
| BUY | retest2 | 2026-04-09 11:30:00 | 1004.05 | 2026-04-10 09:15:00 | 978.25 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-04-09 14:45:00 | 1002.90 | 2026-04-10 09:15:00 | 978.25 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-13 15:00:00 | 996.30 | 2026-04-15 09:15:00 | 1011.20 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-04-28 13:45:00 | 1004.35 | 2026-04-29 09:15:00 | 1013.45 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-28 15:00:00 | 1003.35 | 2026-04-29 09:15:00 | 1013.45 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-04-29 12:45:00 | 1001.40 | 2026-05-06 15:15:00 | 988.95 | STOP_HIT | 1.00 | 1.24% |
