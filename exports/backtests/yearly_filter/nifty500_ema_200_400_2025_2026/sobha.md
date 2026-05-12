# Sobha Ltd. (SOBHA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1425.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 31 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 7 |
| TARGET_HIT | 7 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 25 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 11
- **Target hits / Stop hits / Partials:** 7 / 11 / 7
- **Avg / median % per leg:** 3.07% / 5.00%
- **Sum % (uncompounded):** 76.65%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.34% | -9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.34% | -9.4% |
| SELL (all) | 21 | 14 | 66.7% | 7 | 7 | 7 | 4.10% | 86.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 14 | 66.7% | 7 | 7 | 7 | 4.10% | 86.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 25 | 14 | 56.0% | 7 | 11 | 7 | 3.07% | 76.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 1377.90 | 1268.27 | 1268.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 1390.50 | 1290.98 | 1280.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1487.80 | 1488.48 | 1413.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1487.80 | 1488.48 | 1413.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1544.80 | 1573.19 | 1524.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:30:00 | 1524.80 | 1573.19 | 1524.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1507.80 | 1571.69 | 1524.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1507.80 | 1571.69 | 1524.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1518.30 | 1571.16 | 1524.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 1522.30 | 1571.16 | 1524.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1512.40 | 1570.58 | 1524.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1512.40 | 1570.58 | 1524.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1515.00 | 1570.02 | 1524.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 1506.30 | 1570.02 | 1524.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1518.20 | 1567.37 | 1524.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:15:00 | 1520.80 | 1567.37 | 1524.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1506.90 | 1566.76 | 1524.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1506.90 | 1566.76 | 1524.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1512.00 | 1564.58 | 1524.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 1505.00 | 1564.02 | 1524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1491.70 | 1563.30 | 1523.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:00:00 | 1491.70 | 1563.30 | 1523.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1520.80 | 1557.73 | 1523.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 1522.00 | 1557.34 | 1523.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1521.40 | 1556.98 | 1523.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 1524.90 | 1556.98 | 1523.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1539.30 | 1556.81 | 1523.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 13:30:00 | 1545.80 | 1556.28 | 1523.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1514.10 | 1555.06 | 1523.50 | SL hit (close<static) qty=1.00 sl=1519.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1428.50 | 1505.88 | 1506.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1420.60 | 1505.03 | 1505.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 1496.00 | 1491.08 | 1498.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 14:15:00 | 1496.00 | 1491.08 | 1498.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1496.00 | 1491.08 | 1498.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 15:00:00 | 1496.00 | 1491.08 | 1498.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1491.50 | 1491.08 | 1498.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 1505.00 | 1491.20 | 1498.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1515.30 | 1491.44 | 1498.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:00:00 | 1515.30 | 1491.44 | 1498.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1613.80 | 1504.81 | 1504.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 1627.00 | 1517.83 | 1511.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 1539.20 | 1542.64 | 1526.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 14:15:00 | 1509.30 | 1542.06 | 1527.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1509.30 | 1542.06 | 1527.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1509.30 | 1542.06 | 1527.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1518.50 | 1541.83 | 1527.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1505.50 | 1541.83 | 1527.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1507.00 | 1541.48 | 1526.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1512.40 | 1541.48 | 1526.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1533.20 | 1540.25 | 1526.80 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 1454.00 | 1515.92 | 1516.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 1451.90 | 1515.29 | 1515.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1537.20 | 1503.60 | 1509.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1537.20 | 1503.60 | 1509.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1537.20 | 1503.60 | 1509.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1537.20 | 1503.60 | 1509.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1519.10 | 1503.75 | 1509.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 13:30:00 | 1506.50 | 1503.86 | 1509.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 14:15:00 | 1547.70 | 1505.51 | 1510.34 | SL hit (close>static) qty=1.00 sl=1542.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 11:15:00 | 1580.30 | 1514.49 | 1514.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 10:15:00 | 1601.10 | 1526.19 | 1520.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 1575.70 | 1576.41 | 1551.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:45:00 | 1576.10 | 1576.41 | 1551.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1546.00 | 1577.03 | 1555.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1551.00 | 1577.03 | 1555.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1536.70 | 1576.63 | 1555.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 1536.70 | 1576.63 | 1555.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1532.90 | 1576.19 | 1555.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 1531.50 | 1576.19 | 1555.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1559.50 | 1567.96 | 1553.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 1572.00 | 1567.50 | 1553.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 10:30:00 | 1571.00 | 1567.59 | 1553.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 12:15:00 | 1527.60 | 1565.02 | 1552.74 | SL hit (close<static) qty=1.00 sl=1531.30 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 1425.90 | 1543.59 | 1543.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 12:15:00 | 1404.80 | 1531.59 | 1537.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1502.60 | 1497.67 | 1516.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:45:00 | 1497.10 | 1497.67 | 1516.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1541.00 | 1487.42 | 1506.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:30:00 | 1510.40 | 1497.53 | 1510.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 1570.60 | 1501.44 | 1511.59 | SL hit (close>static) qty=1.00 sl=1568.90 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1430.50 | 1374.82 | 1374.64 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-14 13:30:00 | 1545.80 | 2025-08-18 10:15:00 | 1514.10 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-08-21 11:15:00 | 1544.10 | 2025-08-21 14:15:00 | 1517.60 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-10-15 13:30:00 | 1506.50 | 2025-10-16 14:15:00 | 1547.70 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-10-20 10:30:00 | 1509.20 | 2025-10-20 14:15:00 | 1553.00 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-10-20 11:15:00 | 1501.30 | 2025-10-20 14:15:00 | 1553.00 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-10-23 10:45:00 | 1510.40 | 2025-10-24 11:15:00 | 1538.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-10-24 10:15:00 | 1526.00 | 2025-10-24 14:15:00 | 1536.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-24 14:15:00 | 1524.60 | 2025-10-27 09:15:00 | 1577.00 | STOP_HIT | 1.00 | -3.44% |
| BUY | retest2 | 2025-11-26 14:15:00 | 1572.00 | 2025-11-28 12:15:00 | 1527.60 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2025-11-27 10:30:00 | 1571.00 | 2025-11-28 12:15:00 | 1527.60 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-01-07 12:30:00 | 1510.40 | 2026-01-08 13:15:00 | 1570.60 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1513.90 | 2026-01-20 09:15:00 | 1438.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 1519.80 | 2026-01-20 09:15:00 | 1443.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1490.00 | 2026-01-20 13:15:00 | 1415.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 09:15:00 | 1513.90 | 2026-01-20 14:15:00 | 1367.82 | TARGET_HIT | 0.50 | 9.65% |
| SELL | retest2 | 2026-01-14 10:00:00 | 1519.80 | 2026-01-20 15:15:00 | 1362.51 | TARGET_HIT | 0.50 | 10.35% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1490.00 | 2026-01-20 15:15:00 | 1341.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 1437.40 | 2026-03-02 09:15:00 | 1365.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1440.00 | 2026-03-02 09:15:00 | 1368.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 15:15:00 | 1443.00 | 2026-03-02 09:15:00 | 1370.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:15:00 | 1437.90 | 2026-03-02 09:15:00 | 1366.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-25 11:45:00 | 1437.40 | 2026-03-04 10:15:00 | 1298.70 | TARGET_HIT | 0.50 | 9.65% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1440.00 | 2026-03-09 13:15:00 | 1293.66 | TARGET_HIT | 0.50 | 10.16% |
| SELL | retest2 | 2026-02-25 15:15:00 | 1443.00 | 2026-03-09 13:15:00 | 1296.00 | TARGET_HIT | 0.50 | 10.19% |
| SELL | retest2 | 2026-02-26 13:15:00 | 1437.90 | 2026-03-09 13:15:00 | 1294.11 | TARGET_HIT | 0.50 | 10.00% |
