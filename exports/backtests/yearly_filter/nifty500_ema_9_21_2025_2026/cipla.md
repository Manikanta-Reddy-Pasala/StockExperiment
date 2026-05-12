# Cipla Ltd. (CIPLA)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1348.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 61 |
| ALERT2 | 60 |
| ALERT2_SKIP | 31 |
| ALERT3 | 149 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 55 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 54 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 17 / 41
- **Target hits / Stop hits / Partials:** 1 / 53 / 4
- **Avg / median % per leg:** 0.30% / -0.61%
- **Sum % (uncompounded):** 17.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 3 | 15.0% | 0 | 20 | 0 | -0.54% | -10.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 3 | 15.0% | 0 | 20 | 0 | -0.54% | -10.9% |
| SELL (all) | 38 | 14 | 36.8% | 1 | 33 | 4 | 0.74% | 28.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.71% | -1.7% |
| SELL @ 3rd Alert (retest2) | 37 | 14 | 37.8% | 1 | 32 | 4 | 0.81% | 29.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.71% | -1.7% |
| retest2 (combined) | 57 | 17 | 29.8% | 1 | 52 | 4 | 0.33% | 19.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1508.00 | 1496.50 | 1495.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1511.10 | 1499.42 | 1496.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1503.60 | 1513.99 | 1508.57 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 1490.50 | 1502.99 | 1504.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1477.60 | 1496.26 | 1500.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 1506.00 | 1492.78 | 1497.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 1506.00 | 1492.78 | 1497.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1499.70 | 1494.16 | 1497.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 1506.70 | 1494.16 | 1497.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1494.90 | 1495.41 | 1497.21 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1500.40 | 1498.19 | 1498.15 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 1493.90 | 1498.21 | 1498.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 09:15:00 | 1471.40 | 1489.88 | 1494.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1480.40 | 1471.38 | 1480.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1478.20 | 1471.38 | 1480.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1487.70 | 1474.65 | 1481.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 1487.60 | 1474.65 | 1481.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1485.20 | 1476.76 | 1481.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 1478.20 | 1478.66 | 1481.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1482.20 | 1476.04 | 1477.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 1487.50 | 1479.16 | 1478.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1487.50 | 1479.16 | 1478.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1488.80 | 1484.33 | 1482.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1482.60 | 1484.36 | 1482.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1482.60 | 1484.36 | 1482.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 1480.50 | 1483.59 | 1482.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:45:00 | 1481.30 | 1483.59 | 1482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1480.90 | 1483.05 | 1482.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1477.20 | 1483.05 | 1482.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 1478.10 | 1482.06 | 1482.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 10:15:00 | 1475.20 | 1480.69 | 1481.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 1473.90 | 1472.65 | 1476.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 11:45:00 | 1474.20 | 1472.65 | 1476.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1473.30 | 1472.78 | 1475.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1473.30 | 1472.78 | 1475.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1475.60 | 1473.05 | 1475.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1475.60 | 1473.05 | 1475.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1475.10 | 1473.46 | 1475.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1486.10 | 1473.46 | 1475.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1479.80 | 1474.72 | 1475.73 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 11:15:00 | 1483.10 | 1476.46 | 1476.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 12:15:00 | 1485.40 | 1478.25 | 1477.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1465.10 | 1476.81 | 1476.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1465.10 | 1476.81 | 1476.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1463.90 | 1474.23 | 1475.61 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 1480.90 | 1471.25 | 1470.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1491.80 | 1476.62 | 1474.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1513.10 | 1519.71 | 1514.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 1513.10 | 1519.71 | 1514.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1497.70 | 1515.30 | 1513.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1497.70 | 1515.30 | 1513.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1501.70 | 1512.58 | 1512.15 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 15:15:00 | 1502.80 | 1510.63 | 1511.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 10:15:00 | 1498.10 | 1506.98 | 1509.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1504.70 | 1504.46 | 1507.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1504.70 | 1504.46 | 1507.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 1507.80 | 1505.12 | 1507.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 1526.70 | 1505.12 | 1507.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 1535.50 | 1511.20 | 1509.85 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 1504.00 | 1514.42 | 1515.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1497.00 | 1503.65 | 1508.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1498.60 | 1491.54 | 1497.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:00:00 | 1498.60 | 1491.54 | 1497.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1498.80 | 1493.00 | 1497.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1501.30 | 1493.00 | 1497.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1505.40 | 1495.48 | 1498.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 1505.40 | 1495.48 | 1498.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1500.70 | 1496.52 | 1498.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 1493.00 | 1497.68 | 1498.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 1490.80 | 1494.67 | 1497.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:45:00 | 1495.90 | 1494.53 | 1496.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1512.60 | 1500.24 | 1498.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1512.60 | 1500.24 | 1498.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 1521.50 | 1510.50 | 1506.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 10:15:00 | 1510.30 | 1510.46 | 1506.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 10:45:00 | 1512.00 | 1510.46 | 1506.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1505.90 | 1509.55 | 1506.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:45:00 | 1505.50 | 1509.55 | 1506.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 1506.20 | 1508.88 | 1506.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 1505.20 | 1508.88 | 1506.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1510.00 | 1509.10 | 1506.98 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 1496.90 | 1506.01 | 1506.31 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1506.10 | 1504.73 | 1504.73 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 1503.10 | 1504.57 | 1504.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 1501.20 | 1503.90 | 1504.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1504.00 | 1503.73 | 1504.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 1504.50 | 1503.73 | 1504.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 1507.90 | 1504.56 | 1504.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 1514.80 | 1506.61 | 1505.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1507.00 | 1509.65 | 1507.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:45:00 | 1505.20 | 1509.65 | 1507.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1506.30 | 1508.98 | 1507.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:30:00 | 1506.00 | 1508.98 | 1507.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1507.70 | 1508.72 | 1507.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 1507.70 | 1508.72 | 1507.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1499.10 | 1506.80 | 1506.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1496.10 | 1506.80 | 1506.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 1497.20 | 1504.88 | 1505.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 15:15:00 | 1495.40 | 1502.98 | 1504.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1505.30 | 1503.45 | 1504.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:15:00 | 1504.90 | 1503.45 | 1504.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1503.50 | 1503.46 | 1504.83 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 1510.40 | 1506.49 | 1506.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 1515.00 | 1508.77 | 1507.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1510.40 | 1511.86 | 1509.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 1511.50 | 1511.86 | 1509.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1513.60 | 1512.21 | 1509.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:30:00 | 1513.60 | 1512.21 | 1509.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1510.60 | 1511.86 | 1510.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:45:00 | 1510.00 | 1511.86 | 1510.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1515.00 | 1512.48 | 1510.58 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1493.20 | 1506.70 | 1508.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1481.00 | 1501.56 | 1506.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 11:15:00 | 1495.00 | 1493.11 | 1497.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 12:00:00 | 1495.00 | 1493.11 | 1497.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 1483.90 | 1479.27 | 1483.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 1483.90 | 1479.27 | 1483.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1485.20 | 1480.45 | 1483.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 1485.90 | 1480.45 | 1483.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1486.00 | 1481.56 | 1483.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1478.90 | 1481.56 | 1483.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1483.90 | 1482.03 | 1483.70 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1488.20 | 1485.05 | 1484.72 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 1482.90 | 1484.48 | 1484.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-15 09:15:00 | 1472.20 | 1482.03 | 1483.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 10:15:00 | 1482.50 | 1482.12 | 1483.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 11:00:00 | 1482.50 | 1482.12 | 1483.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1488.50 | 1483.40 | 1483.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1488.50 | 1483.40 | 1483.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1490.30 | 1484.78 | 1484.38 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 1478.30 | 1483.88 | 1484.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 15:15:00 | 1470.70 | 1478.68 | 1481.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1479.40 | 1478.83 | 1481.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 1471.50 | 1479.84 | 1481.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1469.00 | 1475.88 | 1477.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1468.70 | 1474.79 | 1476.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1477.30 | 1471.01 | 1470.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1477.30 | 1471.01 | 1470.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 1482.20 | 1473.45 | 1471.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 1564.80 | 1567.63 | 1552.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-30 10:45:00 | 1562.70 | 1567.63 | 1552.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1553.10 | 1562.81 | 1553.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 14:00:00 | 1553.10 | 1562.81 | 1553.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1560.00 | 1562.25 | 1554.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1562.90 | 1553.75 | 1552.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 1517.30 | 1546.38 | 1549.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1517.30 | 1546.38 | 1549.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1506.70 | 1533.60 | 1542.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1515.40 | 1514.04 | 1525.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1516.70 | 1514.04 | 1525.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1481.90 | 1486.09 | 1495.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:00:00 | 1470.40 | 1482.95 | 1493.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 1502.90 | 1490.16 | 1489.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 1502.90 | 1490.16 | 1489.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 1505.00 | 1496.04 | 1492.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 1559.30 | 1560.08 | 1549.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 11:45:00 | 1559.00 | 1560.08 | 1549.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 1554.20 | 1561.67 | 1557.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 14:00:00 | 1554.20 | 1561.67 | 1557.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1547.80 | 1558.90 | 1556.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1547.80 | 1558.90 | 1556.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 09:15:00 | 1537.20 | 1552.82 | 1553.99 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 11:15:00 | 1566.50 | 1554.99 | 1553.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 13:15:00 | 1573.40 | 1560.75 | 1556.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1584.60 | 1596.39 | 1589.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:00:00 | 1584.60 | 1596.39 | 1589.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1587.30 | 1594.57 | 1588.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 1586.50 | 1594.57 | 1588.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1587.60 | 1593.18 | 1588.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 1587.60 | 1593.18 | 1588.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1576.60 | 1585.68 | 1586.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1571.10 | 1582.76 | 1584.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 1583.40 | 1582.89 | 1584.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 11:00:00 | 1583.40 | 1582.89 | 1584.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1578.90 | 1581.70 | 1583.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:30:00 | 1582.20 | 1581.70 | 1583.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1569.40 | 1578.97 | 1581.99 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 1592.00 | 1584.10 | 1583.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 1597.40 | 1588.04 | 1585.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1587.50 | 1589.24 | 1587.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:00:00 | 1587.50 | 1589.24 | 1587.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1586.70 | 1588.73 | 1587.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1588.00 | 1588.73 | 1587.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1585.10 | 1588.00 | 1586.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1578.30 | 1588.00 | 1586.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1587.70 | 1587.94 | 1587.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1582.00 | 1587.94 | 1587.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1588.10 | 1587.97 | 1587.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:00:00 | 1588.10 | 1587.97 | 1587.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1584.00 | 1587.18 | 1586.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 1584.80 | 1587.18 | 1586.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 12:15:00 | 1580.20 | 1585.78 | 1586.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 13:15:00 | 1567.70 | 1582.17 | 1584.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1589.10 | 1579.44 | 1582.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 1589.10 | 1579.44 | 1582.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1592.00 | 1581.95 | 1583.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1592.00 | 1581.95 | 1583.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 12:15:00 | 1587.60 | 1584.41 | 1584.20 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 14:15:00 | 1578.40 | 1583.26 | 1583.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 1575.00 | 1578.41 | 1580.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 15:15:00 | 1578.90 | 1578.15 | 1580.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 1580.10 | 1578.15 | 1580.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1577.60 | 1578.04 | 1580.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:45:00 | 1581.80 | 1578.04 | 1580.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1568.40 | 1576.11 | 1578.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 1567.60 | 1576.11 | 1578.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 1565.30 | 1554.87 | 1556.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 13:15:00 | 1559.60 | 1557.22 | 1557.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 1559.60 | 1557.22 | 1557.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 14:15:00 | 1564.60 | 1558.69 | 1557.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1558.00 | 1558.98 | 1558.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 1558.00 | 1558.98 | 1558.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 1559.70 | 1559.13 | 1558.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 1558.70 | 1559.13 | 1558.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1561.10 | 1559.92 | 1558.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 1560.10 | 1559.92 | 1558.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1565.10 | 1570.19 | 1566.51 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1554.70 | 1563.88 | 1564.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 14:15:00 | 1548.70 | 1560.84 | 1563.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 1557.40 | 1556.97 | 1559.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 14:00:00 | 1557.40 | 1556.97 | 1559.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1559.40 | 1557.46 | 1559.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 1559.40 | 1557.46 | 1559.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 1555.40 | 1557.05 | 1559.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:15:00 | 1554.00 | 1557.05 | 1559.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 1553.40 | 1556.32 | 1558.65 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 1569.50 | 1559.80 | 1559.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1572.90 | 1565.96 | 1562.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1560.00 | 1568.59 | 1564.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1560.00 | 1568.59 | 1564.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1564.90 | 1567.85 | 1564.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:15:00 | 1567.30 | 1565.76 | 1564.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 13:45:00 | 1566.60 | 1566.11 | 1564.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 11:15:00 | 1555.10 | 1566.64 | 1566.07 | SL hit (close<static) qty=1.00 sl=1558.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 1550.60 | 1563.43 | 1564.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 1541.20 | 1557.30 | 1561.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 1532.10 | 1531.67 | 1540.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:30:00 | 1533.60 | 1531.67 | 1540.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1540.10 | 1534.37 | 1540.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 1540.10 | 1534.37 | 1540.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1538.10 | 1535.12 | 1540.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 15:15:00 | 1535.00 | 1535.12 | 1540.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 1530.20 | 1534.51 | 1538.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1515.40 | 1504.69 | 1504.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1515.40 | 1504.69 | 1504.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1518.30 | 1509.67 | 1507.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1503.00 | 1512.51 | 1510.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1502.60 | 1512.51 | 1510.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1509.40 | 1511.88 | 1510.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 11:45:00 | 1510.10 | 1511.51 | 1510.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:00:00 | 1518.80 | 1514.48 | 1512.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1500.70 | 1511.41 | 1511.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 1500.70 | 1511.41 | 1511.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1495.00 | 1502.60 | 1506.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1508.30 | 1502.44 | 1505.95 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 1512.00 | 1508.49 | 1508.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1520.20 | 1512.26 | 1510.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 1548.20 | 1556.06 | 1545.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 1548.20 | 1556.06 | 1545.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 1551.90 | 1555.23 | 1545.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 1551.90 | 1555.23 | 1545.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 1545.80 | 1553.34 | 1545.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:30:00 | 1539.00 | 1553.34 | 1545.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1541.70 | 1551.01 | 1545.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 1541.70 | 1551.01 | 1545.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1542.10 | 1549.23 | 1545.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:45:00 | 1541.20 | 1549.23 | 1545.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1553.90 | 1550.16 | 1546.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1560.60 | 1549.79 | 1546.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:45:00 | 1556.30 | 1550.93 | 1547.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 13:15:00 | 1554.90 | 1551.24 | 1548.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 1592.00 | 1615.98 | 1616.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 1592.00 | 1615.98 | 1616.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1587.20 | 1610.22 | 1613.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 1575.60 | 1574.59 | 1583.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1584.40 | 1576.55 | 1583.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1584.40 | 1576.55 | 1583.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1584.40 | 1576.55 | 1583.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1581.60 | 1577.56 | 1583.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1583.50 | 1577.56 | 1583.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1583.80 | 1578.81 | 1583.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1582.20 | 1578.81 | 1583.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1577.00 | 1578.45 | 1582.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1568.70 | 1579.27 | 1582.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 1543.50 | 1570.56 | 1576.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1512.00 | 1507.02 | 1506.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1512.00 | 1507.02 | 1506.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 1519.10 | 1514.03 | 1511.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 1518.00 | 1518.10 | 1514.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 14:00:00 | 1518.00 | 1518.10 | 1514.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1520.60 | 1519.31 | 1516.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1516.10 | 1519.31 | 1516.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1522.80 | 1531.46 | 1528.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1522.80 | 1531.46 | 1528.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1526.60 | 1530.49 | 1528.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 1521.90 | 1530.49 | 1528.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1520.10 | 1528.01 | 1527.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 1520.10 | 1528.01 | 1527.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 1526.00 | 1527.61 | 1527.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 1514.40 | 1524.97 | 1526.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1524.00 | 1520.37 | 1523.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1524.00 | 1520.37 | 1523.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1526.50 | 1521.60 | 1523.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 1526.50 | 1521.60 | 1523.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 1526.90 | 1522.66 | 1523.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:00:00 | 1526.90 | 1522.66 | 1523.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 1528.10 | 1523.75 | 1524.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:15:00 | 1526.00 | 1523.75 | 1524.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1531.20 | 1525.84 | 1525.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1531.20 | 1525.84 | 1525.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 11:15:00 | 1532.30 | 1527.13 | 1525.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1517.70 | 1526.35 | 1526.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1517.70 | 1526.35 | 1526.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1521.60 | 1525.40 | 1525.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1512.10 | 1520.73 | 1523.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1523.40 | 1519.47 | 1522.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 1523.40 | 1519.47 | 1522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1521.00 | 1519.77 | 1522.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1518.40 | 1518.66 | 1521.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:45:00 | 1516.80 | 1515.48 | 1517.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 14:15:00 | 1525.00 | 1517.74 | 1516.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 1525.00 | 1517.74 | 1516.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 1527.10 | 1520.77 | 1518.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1522.70 | 1524.09 | 1521.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 1530.00 | 1525.98 | 1523.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1529.40 | 1527.35 | 1525.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1520.80 | 1523.96 | 1524.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1520.80 | 1523.96 | 1524.29 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 1527.70 | 1524.71 | 1524.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 11:15:00 | 1530.90 | 1525.95 | 1525.17 | Break + close above crossover candle high |

### Cycle 50 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1517.00 | 1524.16 | 1524.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 1509.90 | 1519.27 | 1521.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 1510.00 | 1509.74 | 1513.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 1510.00 | 1509.74 | 1513.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1514.50 | 1510.69 | 1513.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1514.50 | 1510.69 | 1513.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1518.30 | 1512.21 | 1514.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:00:00 | 1518.30 | 1512.21 | 1514.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1519.80 | 1513.73 | 1514.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1519.80 | 1513.73 | 1514.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 1517.60 | 1515.19 | 1515.19 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1511.50 | 1516.46 | 1516.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1501.80 | 1511.61 | 1514.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 1502.00 | 1497.92 | 1503.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 1502.00 | 1497.92 | 1503.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1496.80 | 1497.69 | 1502.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:30:00 | 1494.90 | 1496.66 | 1501.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 1494.90 | 1495.00 | 1500.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1510.40 | 1500.89 | 1500.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1510.40 | 1500.89 | 1500.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1512.60 | 1504.71 | 1502.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1502.80 | 1513.04 | 1509.92 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 15:15:00 | 1504.50 | 1508.27 | 1508.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 1501.30 | 1506.87 | 1507.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1507.00 | 1506.90 | 1507.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 1507.00 | 1506.90 | 1507.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 1502.30 | 1505.98 | 1507.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 12:45:00 | 1500.50 | 1504.82 | 1506.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 1500.00 | 1503.86 | 1506.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:15:00 | 1500.40 | 1499.69 | 1500.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 1509.20 | 1501.98 | 1501.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 1509.20 | 1501.98 | 1501.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1516.90 | 1507.26 | 1504.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 1511.20 | 1512.52 | 1507.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 12:00:00 | 1511.20 | 1512.52 | 1507.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1512.70 | 1512.21 | 1508.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1517.80 | 1511.81 | 1509.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 11:15:00 | 1508.00 | 1512.05 | 1509.99 | SL hit (close<static) qty=1.00 sl=1508.20 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 14:15:00 | 1500.80 | 1507.37 | 1508.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 1496.60 | 1503.00 | 1505.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1508.60 | 1499.82 | 1502.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 1508.60 | 1499.82 | 1502.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1506.00 | 1501.06 | 1502.64 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 14:15:00 | 1506.70 | 1503.89 | 1503.62 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1500.50 | 1502.94 | 1503.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 1494.20 | 1499.66 | 1501.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1493.00 | 1489.70 | 1494.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1493.00 | 1489.70 | 1494.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1488.90 | 1489.54 | 1493.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1496.40 | 1489.54 | 1493.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1493.60 | 1490.35 | 1493.92 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1505.30 | 1496.36 | 1496.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1507.10 | 1498.50 | 1497.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 1502.00 | 1503.05 | 1500.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:45:00 | 1503.60 | 1503.05 | 1500.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1501.50 | 1502.74 | 1500.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 1504.30 | 1503.15 | 1501.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 10:00:00 | 1503.10 | 1523.13 | 1519.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 1480.10 | 1514.52 | 1516.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 1480.10 | 1514.52 | 1516.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 1460.30 | 1503.68 | 1511.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 13:15:00 | 1464.50 | 1463.95 | 1473.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 13:45:00 | 1464.70 | 1463.95 | 1473.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1466.50 | 1461.85 | 1466.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1467.00 | 1461.85 | 1466.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1464.00 | 1462.28 | 1466.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1449.20 | 1462.28 | 1466.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1443.50 | 1458.52 | 1464.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1442.10 | 1453.20 | 1460.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1442.30 | 1450.03 | 1458.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1433.00 | 1445.17 | 1453.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1369.99 | 1428.34 | 1440.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 1370.18 | 1428.34 | 1440.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 10:15:00 | 1390.00 | 1389.81 | 1401.97 | SL hit (close>ema200) qty=0.50 sl=1389.81 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 15:15:00 | 1342.90 | 1326.09 | 1324.67 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1306.70 | 1322.21 | 1323.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1295.20 | 1316.81 | 1320.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1313.00 | 1309.50 | 1315.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1313.00 | 1309.50 | 1315.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1332.60 | 1314.92 | 1316.72 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1334.60 | 1318.86 | 1318.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 14:15:00 | 1341.90 | 1333.61 | 1330.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 1341.60 | 1341.94 | 1337.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 14:30:00 | 1340.20 | 1341.94 | 1337.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1337.00 | 1340.80 | 1337.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 1337.00 | 1340.80 | 1337.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1346.30 | 1341.90 | 1338.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 1347.30 | 1341.90 | 1338.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 1348.40 | 1343.89 | 1340.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 10:15:00 | 1332.40 | 1342.23 | 1341.02 | SL hit (close<static) qty=1.00 sl=1336.70 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 11:15:00 | 1332.00 | 1340.18 | 1340.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 1328.80 | 1337.90 | 1339.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1330.10 | 1328.98 | 1333.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1331.90 | 1328.98 | 1333.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 1334.20 | 1330.03 | 1333.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:00:00 | 1334.20 | 1330.03 | 1333.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1330.50 | 1330.12 | 1332.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:45:00 | 1334.70 | 1330.12 | 1332.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1337.70 | 1331.78 | 1333.20 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 1344.60 | 1334.34 | 1334.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 1348.10 | 1337.09 | 1335.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 10:15:00 | 1345.70 | 1346.84 | 1342.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-17 11:00:00 | 1345.70 | 1346.84 | 1342.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1346.10 | 1346.69 | 1342.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 1343.30 | 1346.69 | 1342.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1343.20 | 1346.17 | 1343.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 1343.20 | 1346.17 | 1343.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 1344.10 | 1345.76 | 1343.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:30:00 | 1342.80 | 1345.76 | 1343.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1343.20 | 1345.25 | 1343.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 1349.30 | 1345.25 | 1343.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 1340.60 | 1343.37 | 1342.71 | SL hit (close<static) qty=1.00 sl=1341.50 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1335.10 | 1343.28 | 1343.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1331.00 | 1340.82 | 1342.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1337.40 | 1335.37 | 1338.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 12:00:00 | 1337.40 | 1335.37 | 1338.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1336.40 | 1335.58 | 1338.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 1337.40 | 1335.58 | 1338.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1341.50 | 1336.70 | 1338.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1341.50 | 1336.70 | 1338.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1338.00 | 1336.96 | 1338.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 1320.00 | 1336.96 | 1338.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1341.30 | 1331.71 | 1331.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1341.30 | 1331.71 | 1331.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 1346.80 | 1338.49 | 1335.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1343.80 | 1350.08 | 1344.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:30:00 | 1343.90 | 1350.08 | 1344.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1344.30 | 1348.92 | 1344.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:00:00 | 1347.40 | 1347.53 | 1345.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 1330.90 | 1343.98 | 1344.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1330.90 | 1343.98 | 1344.07 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 1353.20 | 1344.87 | 1344.18 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 1331.20 | 1343.11 | 1343.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1323.70 | 1337.20 | 1340.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 1324.00 | 1323.08 | 1329.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:30:00 | 1323.50 | 1323.08 | 1329.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1323.80 | 1322.44 | 1328.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1323.80 | 1322.44 | 1328.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1328.40 | 1324.44 | 1328.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1318.40 | 1324.40 | 1327.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1304.50 | 1324.76 | 1326.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 12:45:00 | 1321.00 | 1317.13 | 1321.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 14:30:00 | 1321.00 | 1319.32 | 1321.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1325.00 | 1320.45 | 1322.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1327.60 | 1320.45 | 1322.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1322.50 | 1320.86 | 1322.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 1325.50 | 1320.86 | 1322.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1326.20 | 1321.93 | 1322.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 11:00:00 | 1326.20 | 1321.93 | 1322.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-10 11:15:00 | 1331.90 | 1323.92 | 1323.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1331.90 | 1323.92 | 1323.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 12:15:00 | 1337.70 | 1326.68 | 1324.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 1330.00 | 1330.07 | 1327.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1335.60 | 1330.17 | 1328.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:45:00 | 1335.60 | 1330.94 | 1329.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 1325.10 | 1328.37 | 1328.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 1325.10 | 1328.37 | 1328.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1310.30 | 1324.66 | 1326.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 15:15:00 | 1316.90 | 1316.29 | 1320.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 09:15:00 | 1319.30 | 1316.29 | 1320.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 1310.30 | 1315.10 | 1319.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 1298.60 | 1315.10 | 1319.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 13:15:00 | 1233.67 | 1243.71 | 1252.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1235.80 | 1228.19 | 1236.16 | SL hit (close>ema200) qty=0.50 sl=1228.19 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 14:15:00 | 1243.60 | 1240.06 | 1239.70 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1235.30 | 1240.15 | 1240.17 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1242.30 | 1240.58 | 1240.37 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1226.70 | 1238.03 | 1239.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 1209.90 | 1226.11 | 1231.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 15:15:00 | 1195.00 | 1193.80 | 1204.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-06 09:15:00 | 1177.40 | 1193.80 | 1204.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1197.50 | 1189.29 | 1197.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 1197.50 | 1189.29 | 1197.35 | SL hit (close>ema400) qty=1.00 sl=1197.35 alert=retest1 |

### Cycle 77 — BUY (started 2026-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 14:15:00 | 1202.00 | 1199.94 | 1199.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 1206.60 | 1201.60 | 1200.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 1230.10 | 1230.26 | 1223.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:30:00 | 1228.20 | 1230.26 | 1223.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1211.70 | 1226.47 | 1223.01 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1211.90 | 1219.98 | 1220.60 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 1223.10 | 1219.89 | 1219.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 13:15:00 | 1225.60 | 1221.77 | 1220.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 10:15:00 | 1235.90 | 1237.50 | 1233.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 11:00:00 | 1235.90 | 1237.50 | 1233.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 1234.70 | 1236.94 | 1233.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:00:00 | 1234.70 | 1236.94 | 1233.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 1231.00 | 1235.75 | 1233.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 1231.00 | 1235.75 | 1233.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1231.70 | 1234.94 | 1233.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 1231.70 | 1234.94 | 1233.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1228.00 | 1232.78 | 1232.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 1229.10 | 1232.78 | 1232.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1230.20 | 1232.57 | 1232.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 1230.90 | 1232.57 | 1232.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1230.80 | 1232.22 | 1232.25 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 13:15:00 | 1233.70 | 1232.41 | 1232.33 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 1226.80 | 1231.37 | 1231.88 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 1235.10 | 1232.37 | 1232.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 1289.30 | 1244.99 | 1238.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1312.20 | 1312.47 | 1299.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:00:00 | 1312.20 | 1312.47 | 1299.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1300.30 | 1309.03 | 1301.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:30:00 | 1300.10 | 1309.03 | 1301.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1307.40 | 1308.71 | 1301.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-28 15:15:00 | 1310.90 | 1308.71 | 1301.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 1312.00 | 1313.44 | 1311.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-21 13:45:00 | 1478.20 | 2025-05-23 12:15:00 | 1487.50 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-05-23 11:15:00 | 1482.20 | 2025-05-23 12:15:00 | 1487.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-06-20 15:15:00 | 1493.00 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1490.80 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-23 12:45:00 | 1495.90 | 2025-06-24 09:15:00 | 1512.60 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-07-18 10:15:00 | 1471.50 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1469.00 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1468.70 | 2025-07-24 09:15:00 | 1477.30 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1562.90 | 2025-08-01 09:15:00 | 1517.30 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-08-07 11:00:00 | 1470.40 | 2025-08-11 10:15:00 | 1502.90 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-05 11:15:00 | 1567.60 | 2025-09-10 13:15:00 | 1559.60 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2025-09-10 10:30:00 | 1565.30 | 2025-09-10 13:15:00 | 1559.60 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2025-09-19 13:15:00 | 1567.30 | 2025-09-22 11:15:00 | 1555.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-09-19 13:45:00 | 1566.60 | 2025-09-22 11:15:00 | 1555.10 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-24 15:15:00 | 1535.00 | 2025-10-01 13:15:00 | 1515.40 | STOP_HIT | 1.00 | 1.28% |
| SELL | retest2 | 2025-09-25 10:30:00 | 1530.20 | 2025-10-01 13:15:00 | 1515.40 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-10-06 11:45:00 | 1510.10 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-07 13:00:00 | 1518.80 | 2025-10-08 09:15:00 | 1500.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-15 09:15:00 | 1560.60 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.01% |
| BUY | retest2 | 2025-10-15 10:45:00 | 1556.30 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2025-10-15 13:15:00 | 1554.90 | 2025-10-24 12:15:00 | 1592.00 | STOP_HIT | 1.00 | 2.39% |
| SELL | retest2 | 2025-10-30 09:15:00 | 1568.70 | 2025-11-10 12:15:00 | 1512.00 | STOP_HIT | 1.00 | 3.61% |
| SELL | retest2 | 2025-10-30 13:45:00 | 1543.50 | 2025-11-10 12:15:00 | 1512.00 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2025-11-19 15:15:00 | 1526.00 | 2025-11-20 10:15:00 | 1531.20 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1518.40 | 2025-11-26 14:15:00 | 1525.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-11-25 12:45:00 | 1516.80 | 2025-11-26 14:15:00 | 1525.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-11-28 13:00:00 | 1530.00 | 2025-12-02 09:15:00 | 1520.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-12-01 12:15:00 | 1529.40 | 2025-12-02 09:15:00 | 1520.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-12-10 11:30:00 | 1494.90 | 2025-12-11 12:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-10 13:30:00 | 1494.90 | 2025-12-11 12:15:00 | 1510.40 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-12-16 12:45:00 | 1500.50 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-16 14:00:00 | 1500.00 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-12-18 13:15:00 | 1500.40 | 2025-12-19 10:15:00 | 1509.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-23 09:15:00 | 1517.80 | 2025-12-23 11:15:00 | 1508.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-01-02 09:30:00 | 1504.30 | 2026-01-07 10:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-01-07 10:00:00 | 1503.10 | 2026-01-07 10:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1442.10 | 2026-01-16 09:15:00 | 1369.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1442.30 | 2026-01-16 09:15:00 | 1370.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1442.10 | 2026-01-20 10:15:00 | 1390.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1442.30 | 2026-01-20 10:15:00 | 1390.00 | STOP_HIT | 0.50 | 3.63% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1433.00 | 2026-01-23 13:15:00 | 1361.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1433.00 | 2026-01-27 09:15:00 | 1289.70 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-11 11:15:00 | 1347.30 | 2026-02-12 10:15:00 | 1332.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-11 12:45:00 | 1348.40 | 2026-02-12 10:15:00 | 1332.40 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-18 09:15:00 | 1349.30 | 2026-02-18 11:15:00 | 1340.60 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-02-18 14:30:00 | 1346.50 | 2026-02-19 12:15:00 | 1338.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-02-23 09:15:00 | 1320.00 | 2026-02-25 10:15:00 | 1341.30 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-27 14:00:00 | 1347.40 | 2026-03-02 09:15:00 | 1330.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1318.40 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1304.50 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-03-09 12:45:00 | 1321.00 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-09 14:30:00 | 1321.00 | 2026-03-10 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1335.60 | 2026-03-12 14:15:00 | 1325.10 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-03-12 10:45:00 | 1335.60 | 2026-03-12 14:15:00 | 1325.10 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-03-16 10:15:00 | 1298.60 | 2026-03-23 13:15:00 | 1233.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 10:15:00 | 1298.60 | 2026-03-25 09:15:00 | 1235.80 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest1 | 2026-04-06 09:15:00 | 1177.40 | 2026-04-06 13:15:00 | 1197.50 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1193.40 | 2026-04-07 14:15:00 | 1202.00 | STOP_HIT | 1.00 | -0.72% |
