# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2026-01-19 09:15:00 → 2026-05-08 15:15:00 (518 bars)
- **Last close:** 1647.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 32 |
| ALERT1 | 18 |
| ALERT2 | 17 |
| ALERT2_SKIP | 8 |
| ALERT3 | 50 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 3 |
| TARGET_HIT | 0 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 32 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 23
- **Target hits / Stop hits / Partials:** 0 / 29 / 3
- **Avg / median % per leg:** -0.30% / -0.94%
- **Sum % (uncompounded):** -9.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 3 | 21.4% | 0 | 14 | 0 | -0.91% | -12.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 0 | 14 | 0 | -0.91% | -12.8% |
| SELL (all) | 18 | 6 | 33.3% | 0 | 15 | 3 | 0.18% | 3.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 0 | 15 | 3 | 0.18% | 3.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 32 | 9 | 28.1% | 0 | 29 | 3 | -0.30% | -9.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1693.60 | 1660.65 | 1659.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 1703.40 | 1679.32 | 1669.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 1691.40 | 1693.27 | 1680.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 10:15:00 | 1691.40 | 1693.27 | 1680.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1691.40 | 1693.27 | 1680.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 1691.40 | 1693.27 | 1680.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1666.20 | 1687.85 | 1679.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1666.20 | 1687.85 | 1679.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1650.30 | 1680.34 | 1676.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1650.30 | 1680.34 | 1676.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 1635.60 | 1671.39 | 1672.88 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 15:15:00 | 1712.00 | 1676.73 | 1674.88 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1641.10 | 1669.61 | 1671.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 10:15:00 | 1627.20 | 1661.12 | 1667.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1637.50 | 1636.93 | 1652.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 15:00:00 | 1637.50 | 1636.93 | 1652.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1643.50 | 1631.75 | 1644.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:00:00 | 1643.50 | 1631.75 | 1644.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 12:15:00 | 1641.30 | 1633.66 | 1644.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 12:45:00 | 1651.30 | 1633.66 | 1644.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 13:15:00 | 1630.60 | 1633.04 | 1642.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 09:15:00 | 1614.70 | 1639.03 | 1641.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 1653.10 | 1631.30 | 1635.09 | SL hit (close>static) qty=1.00 sl=1643.30 alert=retest2 |

### Cycle 5 — BUY (started 2026-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 15:15:00 | 1648.60 | 1638.01 | 1637.68 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1610.10 | 1634.32 | 1636.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1594.30 | 1620.02 | 1627.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 12:15:00 | 1443.00 | 1438.07 | 1470.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:00:00 | 1443.00 | 1438.07 | 1470.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 1583.80 | 1467.21 | 1481.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 1583.80 | 1467.21 | 1481.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 1559.20 | 1485.61 | 1488.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 1540.00 | 1485.61 | 1488.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 15:15:00 | 1540.00 | 1496.49 | 1492.95 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 1487.00 | 1512.88 | 1515.80 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 10:15:00 | 1545.90 | 1521.12 | 1517.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 11:15:00 | 1546.70 | 1526.23 | 1520.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1533.30 | 1540.29 | 1531.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:15:00 | 1558.70 | 1541.78 | 1533.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 14:45:00 | 1551.70 | 1548.30 | 1538.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 1551.20 | 1544.44 | 1539.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1525.00 | 1535.46 | 1536.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1509.60 | 1530.29 | 1534.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 14:15:00 | 1496.10 | 1495.40 | 1507.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 15:00:00 | 1496.10 | 1495.40 | 1507.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1507.00 | 1496.94 | 1505.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 1510.80 | 1496.94 | 1505.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 1508.00 | 1499.15 | 1505.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:30:00 | 1505.60 | 1499.15 | 1505.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1499.00 | 1499.12 | 1505.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 12:15:00 | 1497.30 | 1499.12 | 1505.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 1493.50 | 1501.96 | 1504.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:00:00 | 1496.50 | 1500.86 | 1504.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:15:00 | 1495.90 | 1500.97 | 1503.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1486.00 | 1497.98 | 1502.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:15:00 | 1481.60 | 1495.10 | 1500.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:45:00 | 1477.60 | 1491.84 | 1498.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1520.80 | 1497.51 | 1496.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1546.20 | 1511.98 | 1504.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1493.80 | 1519.09 | 1511.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 1493.80 | 1519.09 | 1511.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1493.60 | 1513.99 | 1509.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:30:00 | 1492.00 | 1513.99 | 1509.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1488.70 | 1506.20 | 1506.70 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1525.10 | 1506.47 | 1504.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 1530.20 | 1511.21 | 1507.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1513.70 | 1514.14 | 1509.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 1513.70 | 1514.14 | 1509.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1514.70 | 1515.21 | 1510.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:30:00 | 1527.00 | 1517.40 | 1512.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1523.70 | 1519.09 | 1514.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:45:00 | 1521.40 | 1519.82 | 1514.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 1492.60 | 1515.18 | 1514.13 | SL hit (close<static) qty=1.00 sl=1507.10 alert=retest2 |

### Cycle 14 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1482.70 | 1508.69 | 1511.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 1478.80 | 1498.68 | 1506.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 1474.00 | 1469.98 | 1482.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 09:30:00 | 1473.50 | 1469.48 | 1480.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 1472.20 | 1468.78 | 1478.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:30:00 | 1476.50 | 1468.78 | 1478.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 1482.60 | 1471.54 | 1478.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:00:00 | 1482.60 | 1471.54 | 1478.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 1482.60 | 1473.75 | 1479.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 13:30:00 | 1482.00 | 1473.75 | 1479.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1475.80 | 1475.87 | 1478.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 1466.90 | 1475.87 | 1478.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1467.70 | 1471.87 | 1475.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:15:00 | 1466.40 | 1471.50 | 1475.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.56 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1394.32 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1393.08 | 1426.88 | 1445.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 11:15:00 | 1423.70 | 1420.84 | 1439.32 | SL hit (close>ema200) qty=0.50 sl=1420.84 alert=retest2 |

### Cycle 15 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1464.90 | 1443.09 | 1442.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1470.00 | 1452.04 | 1446.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 1467.60 | 1468.88 | 1458.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 12:00:00 | 1467.60 | 1468.88 | 1458.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1460.70 | 1468.22 | 1460.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1460.70 | 1468.22 | 1460.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1456.50 | 1465.88 | 1460.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1449.60 | 1465.88 | 1460.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1452.30 | 1463.16 | 1459.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 1457.70 | 1463.16 | 1459.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 1463.50 | 1461.78 | 1459.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:45:00 | 1457.90 | 1461.00 | 1459.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 10:15:00 | 1449.80 | 1457.93 | 1458.71 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 1477.40 | 1458.13 | 1456.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 1497.00 | 1467.80 | 1461.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1509.50 | 1517.71 | 1500.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 1508.80 | 1517.71 | 1500.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1507.60 | 1513.67 | 1501.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1507.60 | 1513.67 | 1501.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 1497.30 | 1510.40 | 1501.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 1497.30 | 1510.40 | 1501.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1468.20 | 1501.96 | 1498.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:45:00 | 1480.00 | 1501.96 | 1498.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1505.50 | 1504.14 | 1500.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:30:00 | 1505.00 | 1504.14 | 1500.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1497.20 | 1503.18 | 1500.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1497.20 | 1503.18 | 1500.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1504.00 | 1503.34 | 1501.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1472.70 | 1503.34 | 1501.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 1456.30 | 1493.93 | 1497.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 1449.80 | 1485.11 | 1492.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 1454.60 | 1451.89 | 1467.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 1454.60 | 1451.89 | 1467.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1466.60 | 1454.83 | 1467.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1466.60 | 1454.83 | 1467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1471.30 | 1458.13 | 1467.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 1478.40 | 1458.13 | 1467.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1471.20 | 1460.74 | 1468.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:45:00 | 1472.60 | 1460.74 | 1468.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1461.30 | 1460.85 | 1467.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1461.60 | 1460.85 | 1467.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1460.20 | 1460.72 | 1466.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1496.90 | 1460.72 | 1466.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1480.00 | 1464.58 | 1468.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 10:45:00 | 1474.30 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:15:00 | 1472.40 | 1467.16 | 1468.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1483.00 | 1471.58 | 1470.73 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1419.70 | 1461.52 | 1466.53 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 1469.60 | 1452.06 | 1451.82 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 14:15:00 | 1431.40 | 1448.68 | 1450.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1396.40 | 1435.39 | 1443.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 14:15:00 | 1429.00 | 1423.94 | 1433.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 15:00:00 | 1429.00 | 1423.94 | 1433.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 1425.10 | 1424.17 | 1432.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 1415.30 | 1424.17 | 1432.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1443.60 | 1428.06 | 1433.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:00:00 | 1443.60 | 1428.06 | 1433.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1437.20 | 1429.89 | 1434.22 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 1463.70 | 1441.73 | 1438.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 1469.60 | 1447.30 | 1441.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1442.20 | 1449.91 | 1444.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:30:00 | 1467.20 | 1448.52 | 1444.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 14:30:00 | 1460.40 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:00:00 | 1461.10 | 1451.43 | 1446.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1492.20 | 1452.58 | 1447.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 09:15:00 | 1481.70 | 1488.67 | 1474.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 10:00:00 | 1481.70 | 1488.67 | 1474.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1465.70 | 1498.34 | 1493.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1465.00 | 1486.53 | 1488.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 1462.40 | 1481.71 | 1486.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1480.20 | 1471.04 | 1478.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 12:30:00 | 1471.60 | 1474.89 | 1478.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 1472.10 | 1474.75 | 1478.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1506.20 | 1484.69 | 1482.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 13:15:00 | 1528.60 | 1499.42 | 1490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 1605.20 | 1606.70 | 1580.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 1596.40 | 1606.70 | 1580.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1596.70 | 1604.70 | 1582.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 1622.30 | 1607.92 | 1587.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1661.10 | 1670.51 | 1670.75 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 1698.40 | 1675.73 | 1673.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 1711.00 | 1682.79 | 1676.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1684.60 | 1689.76 | 1681.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1684.60 | 1689.76 | 1681.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1683.50 | 1688.51 | 1681.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 1686.00 | 1688.51 | 1681.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1680.50 | 1686.91 | 1681.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1672.50 | 1686.91 | 1681.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1654.50 | 1680.43 | 1679.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 1654.50 | 1680.43 | 1679.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1650.00 | 1674.34 | 1676.69 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2026-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 11:15:00 | 1673.90 | 1670.87 | 1670.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1679.40 | 1672.84 | 1671.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1668.10 | 1673.52 | 1672.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 1668.10 | 1673.52 | 1672.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1712.40 | 1681.29 | 1675.93 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 10:15:00 | 1645.50 | 1677.57 | 1678.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2026-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 14:15:00 | 1683.50 | 1679.24 | 1679.08 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 1662.80 | 1676.88 | 1678.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 10:15:00 | 1657.60 | 1673.02 | 1676.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-30 09:15:00 | 1614.70 | 2026-01-30 13:15:00 | 1653.10 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-02-05 15:15:00 | 1540.00 | 2026-02-05 15:15:00 | 1540.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-12 12:15:00 | 1558.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1551.70 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-13 11:45:00 | 1551.20 | 2026-02-13 15:15:00 | 1525.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2026-02-18 12:15:00 | 1497.30 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-19 09:15:00 | 1493.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-19 10:00:00 | 1496.50 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-19 11:15:00 | 1495.90 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-02-19 13:15:00 | 1481.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-02-19 13:45:00 | 1477.60 | 2026-02-20 13:15:00 | 1520.80 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-02-26 10:30:00 | 1527.00 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-26 12:45:00 | 1523.70 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-02-26 13:45:00 | 1521.40 | 2026-02-27 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 09:15:00 | 1393.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 09:15:00 | 1394.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 09:15:00 | 1393.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 10:15:00 | 1466.90 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1467.70 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 3.00% |
| SELL | retest2 | 2026-03-05 15:15:00 | 1466.40 | 2026-03-09 11:15:00 | 1423.70 | STOP_HIT | 0.50 | 2.91% |
| BUY | retest2 | 2026-03-12 10:15:00 | 1457.70 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-03-12 11:45:00 | 1463.50 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-03-12 12:45:00 | 1457.90 | 2026-03-13 10:15:00 | 1449.80 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-25 10:45:00 | 1474.30 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-03-25 11:15:00 | 1472.40 | 2026-03-25 12:15:00 | 1483.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-07 12:30:00 | 1467.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-07 14:30:00 | 1460.40 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2026-04-07 15:00:00 | 1461.10 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1492.20 | 2026-04-13 11:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-04-15 12:30:00 | 1471.60 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-15 13:30:00 | 1472.10 | 2026-04-16 10:15:00 | 1506.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2026-04-21 11:45:00 | 1622.30 | 2026-04-28 13:15:00 | 1661.10 | STOP_HIT | 1.00 | 2.39% |
