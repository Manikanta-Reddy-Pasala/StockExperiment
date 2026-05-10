# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 25 |
| ALERT3 | 154 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 67 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 75 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 45 / 30
- **Target hits / Stop hits / Partials:** 0 / 69 / 6
- **Avg / median % per leg:** 1.37% / 1.05%
- **Sum % (uncompounded):** 102.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 34 | 75.6% | 0 | 44 | 1 | 1.76% | 79.0% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.16% | 0.8% |
| BUY @ 3rd Alert (retest2) | 40 | 32 | 80.0% | 0 | 40 | 0 | 1.96% | 78.2% |
| SELL (all) | 30 | 11 | 36.7% | 0 | 25 | 5 | 0.78% | 23.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 11 | 36.7% | 0 | 25 | 5 | 0.78% | 23.4% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.16% | 0.8% |
| retest2 (combined) | 70 | 43 | 61.4% | 0 | 65 | 5 | 1.45% | 101.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1531.30 | 1504.76 | 1502.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1537.00 | 1516.80 | 1508.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1533.10 | 1538.03 | 1530.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 1527.80 | 1538.03 | 1530.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1532.10 | 1536.84 | 1530.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:30:00 | 1515.40 | 1536.84 | 1530.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1530.40 | 1535.55 | 1530.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 1530.40 | 1535.55 | 1530.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1537.60 | 1535.96 | 1531.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1546.30 | 1538.03 | 1532.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:15:00 | 1549.80 | 1544.96 | 1538.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1590.00 | 1597.88 | 1598.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 15:15:00 | 1590.00 | 1597.88 | 1598.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1590.00 | 1597.88 | 1598.76 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1607.40 | 1599.78 | 1599.54 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1587.30 | 1597.29 | 1598.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1573.10 | 1592.45 | 1596.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 12:15:00 | 1593.00 | 1592.56 | 1595.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 13:00:00 | 1593.00 | 1592.56 | 1595.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1600.00 | 1594.05 | 1596.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1600.00 | 1594.05 | 1596.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1598.50 | 1594.94 | 1596.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 1600.00 | 1594.94 | 1596.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1598.00 | 1595.55 | 1596.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1607.70 | 1595.55 | 1596.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1599.50 | 1596.34 | 1596.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 1601.00 | 1596.34 | 1596.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1607.30 | 1598.53 | 1597.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 1615.40 | 1604.09 | 1600.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1593.10 | 1602.45 | 1600.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:30:00 | 1590.20 | 1602.45 | 1600.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 10:15:00 | 1599.10 | 1601.78 | 1600.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 10:30:00 | 1596.90 | 1601.78 | 1600.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 11:15:00 | 1599.70 | 1601.37 | 1600.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 12:30:00 | 1604.50 | 1600.95 | 1600.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 13:15:00 | 1596.00 | 1599.96 | 1599.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 13:15:00 | 1596.00 | 1599.96 | 1599.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 1591.60 | 1598.29 | 1599.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 1567.10 | 1555.71 | 1567.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 1567.10 | 1555.71 | 1567.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 1582.60 | 1561.09 | 1568.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:00:00 | 1582.60 | 1561.09 | 1568.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 1578.30 | 1564.53 | 1569.35 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 1589.60 | 1575.42 | 1573.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 12:15:00 | 1596.90 | 1579.44 | 1576.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1583.40 | 1593.08 | 1584.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 1584.00 | 1593.08 | 1584.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1581.20 | 1590.71 | 1584.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 1581.20 | 1590.71 | 1584.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1578.00 | 1588.17 | 1583.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 1575.50 | 1588.17 | 1583.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1578.00 | 1584.57 | 1582.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 13:30:00 | 1573.90 | 1584.57 | 1582.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1579.60 | 1584.90 | 1583.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:45:00 | 1581.30 | 1584.90 | 1583.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 1591.50 | 1586.22 | 1584.30 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1569.60 | 1583.00 | 1583.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1562.20 | 1578.84 | 1581.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1576.50 | 1576.17 | 1579.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 1572.30 | 1576.17 | 1579.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1575.00 | 1575.94 | 1579.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 1577.80 | 1575.94 | 1579.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1589.10 | 1578.57 | 1580.10 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 13:15:00 | 1590.80 | 1582.86 | 1581.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1595.90 | 1585.47 | 1583.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1626.50 | 1631.74 | 1618.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:45:00 | 1622.50 | 1631.74 | 1618.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1644.70 | 1648.06 | 1633.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1640.80 | 1648.06 | 1633.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1627.80 | 1644.32 | 1635.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1627.80 | 1644.32 | 1635.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1619.00 | 1639.25 | 1634.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1619.00 | 1639.25 | 1634.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1614.30 | 1629.34 | 1630.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 15:15:00 | 1600.00 | 1616.15 | 1623.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1599.80 | 1598.92 | 1609.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1599.80 | 1598.92 | 1609.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1599.60 | 1596.28 | 1604.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 1602.20 | 1596.28 | 1604.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1613.10 | 1600.00 | 1604.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1613.10 | 1600.00 | 1604.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1615.70 | 1603.14 | 1605.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1619.80 | 1603.14 | 1605.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 1626.80 | 1609.96 | 1608.65 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 1598.00 | 1612.57 | 1613.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1580.20 | 1602.70 | 1608.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1587.10 | 1576.22 | 1588.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:30:00 | 1587.80 | 1576.22 | 1588.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1605.00 | 1581.98 | 1590.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1605.00 | 1581.98 | 1590.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1606.20 | 1586.82 | 1591.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1605.90 | 1586.82 | 1591.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1617.70 | 1597.74 | 1595.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 1623.80 | 1605.52 | 1599.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 11:15:00 | 1615.60 | 1618.79 | 1612.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 12:15:00 | 1608.40 | 1616.71 | 1611.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 1608.40 | 1616.71 | 1611.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 1608.40 | 1616.71 | 1611.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1590.20 | 1611.41 | 1609.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1590.20 | 1611.41 | 1609.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 1600.00 | 1607.75 | 1608.49 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1621.50 | 1610.36 | 1608.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 14:15:00 | 1629.90 | 1618.28 | 1613.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1618.00 | 1618.23 | 1614.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 09:15:00 | 1626.70 | 1618.23 | 1614.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1610.90 | 1616.76 | 1613.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1610.90 | 1616.76 | 1613.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1605.40 | 1614.49 | 1613.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1605.40 | 1614.49 | 1613.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1592.90 | 1610.17 | 1611.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1582.10 | 1604.56 | 1608.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 10:15:00 | 1587.20 | 1586.93 | 1596.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 11:00:00 | 1587.20 | 1586.93 | 1596.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1537.50 | 1517.16 | 1525.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 1537.90 | 1517.16 | 1525.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1540.90 | 1521.91 | 1526.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 1548.30 | 1521.91 | 1526.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1531.10 | 1525.03 | 1527.25 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1542.70 | 1530.40 | 1529.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 1546.80 | 1535.36 | 1531.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 15:15:00 | 1543.40 | 1545.46 | 1539.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1559.90 | 1545.46 | 1539.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1517.40 | 1554.93 | 1550.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1517.40 | 1554.93 | 1550.40 | SL hit (close<ema400) qty=1.00 sl=1550.40 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1514.90 | 1554.93 | 1550.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1523.20 | 1548.58 | 1547.92 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 1515.50 | 1541.97 | 1544.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 1504.00 | 1521.78 | 1532.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1504.40 | 1497.55 | 1507.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 09:30:00 | 1504.80 | 1497.55 | 1507.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1502.10 | 1498.46 | 1506.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 1501.30 | 1500.43 | 1506.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 1501.50 | 1500.64 | 1506.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 1512.20 | 1503.65 | 1506.72 | SL hit (close>static) qty=1.00 sl=1511.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 1512.20 | 1503.65 | 1506.72 | SL hit (close>static) qty=1.00 sl=1511.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 1501.20 | 1504.13 | 1506.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 1499.90 | 1489.61 | 1493.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1511.10 | 1493.91 | 1494.72 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 1511.10 | 1493.91 | 1494.72 | SL hit (close>static) qty=1.00 sl=1511.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 1511.10 | 1493.91 | 1494.72 | SL hit (close>static) qty=1.00 sl=1511.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1511.10 | 1493.91 | 1494.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 11:15:00 | 1503.50 | 1495.83 | 1495.51 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1489.60 | 1496.31 | 1496.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1486.10 | 1493.15 | 1494.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1491.10 | 1490.05 | 1492.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1492.50 | 1490.05 | 1492.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1487.00 | 1489.44 | 1492.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1490.00 | 1489.44 | 1492.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1492.70 | 1490.09 | 1492.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1492.70 | 1490.09 | 1492.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1487.20 | 1489.51 | 1491.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:30:00 | 1491.50 | 1489.51 | 1491.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1492.40 | 1490.09 | 1491.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:30:00 | 1491.40 | 1490.09 | 1491.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 14:15:00 | 1492.40 | 1490.55 | 1491.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 14:30:00 | 1492.40 | 1490.55 | 1491.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1495.90 | 1491.62 | 1492.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1481.80 | 1491.62 | 1492.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1470.60 | 1487.42 | 1490.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:30:00 | 1465.40 | 1478.19 | 1484.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 1461.00 | 1472.09 | 1479.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1522.60 | 1464.64 | 1464.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 1522.60 | 1464.64 | 1464.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 1522.60 | 1464.64 | 1464.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 11:15:00 | 1537.90 | 1489.43 | 1476.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 1506.60 | 1511.49 | 1494.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:45:00 | 1524.60 | 1513.39 | 1497.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 11:15:00 | 1524.10 | 1513.39 | 1497.02 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1492.60 | 1508.94 | 1502.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1492.60 | 1508.94 | 1502.44 | SL hit (close<ema400) qty=1.00 sl=1502.44 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1492.60 | 1508.94 | 1502.44 | SL hit (close<ema400) qty=1.00 sl=1502.44 alert=retest1 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 15:15:00 | 1511.50 | 1505.17 | 1502.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:15:00 | 1513.90 | 1506.11 | 1503.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 11:45:00 | 1511.10 | 1505.95 | 1503.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 1512.20 | 1503.74 | 1503.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 15:15:00 | 1499.00 | 1502.79 | 1502.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 1477.20 | 1497.67 | 1500.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1465.90 | 1462.37 | 1471.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1465.90 | 1462.37 | 1471.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1473.90 | 1464.68 | 1471.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 1480.00 | 1464.68 | 1471.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1473.10 | 1466.36 | 1471.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1475.90 | 1466.36 | 1471.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1471.90 | 1467.20 | 1470.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 1471.90 | 1467.20 | 1470.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1475.60 | 1468.88 | 1471.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1475.60 | 1468.88 | 1471.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1470.70 | 1469.25 | 1471.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:30:00 | 1475.00 | 1469.25 | 1471.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1472.50 | 1469.90 | 1471.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:45:00 | 1474.20 | 1469.90 | 1471.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1479.00 | 1471.72 | 1472.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 15:00:00 | 1479.00 | 1471.72 | 1472.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1470.20 | 1471.41 | 1471.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 1456.00 | 1471.41 | 1471.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 09:15:00 | 1453.30 | 1439.18 | 1438.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 09:15:00 | 1453.30 | 1439.18 | 1438.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1461.30 | 1443.60 | 1440.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1435.90 | 1445.45 | 1442.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 1435.90 | 1445.45 | 1442.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1432.00 | 1442.76 | 1441.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1434.70 | 1441.15 | 1441.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 10:15:00 | 1422.00 | 1437.32 | 1439.27 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1481.00 | 1442.21 | 1439.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 12:15:00 | 1491.00 | 1463.60 | 1451.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1564.00 | 1567.68 | 1545.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 1564.00 | 1567.68 | 1545.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1574.60 | 1583.43 | 1573.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 1569.50 | 1583.43 | 1573.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1570.40 | 1580.82 | 1573.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1570.40 | 1580.82 | 1573.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1571.40 | 1578.94 | 1573.42 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 15:15:00 | 1557.00 | 1568.60 | 1569.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 1539.50 | 1562.78 | 1567.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1507.40 | 1507.10 | 1521.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:45:00 | 1512.10 | 1507.10 | 1521.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1518.00 | 1509.28 | 1521.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1518.00 | 1509.28 | 1521.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 1517.10 | 1512.33 | 1520.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 12:30:00 | 1519.00 | 1512.33 | 1520.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1516.90 | 1513.77 | 1520.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 1519.40 | 1513.77 | 1520.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1517.30 | 1514.48 | 1519.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1570.10 | 1514.48 | 1519.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1566.90 | 1524.96 | 1524.16 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 1513.80 | 1533.17 | 1535.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 11:15:00 | 1495.50 | 1515.59 | 1525.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1513.90 | 1502.95 | 1510.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 1513.90 | 1502.95 | 1510.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 1518.60 | 1506.08 | 1511.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 1519.60 | 1506.08 | 1511.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 1527.80 | 1515.95 | 1515.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 11:15:00 | 1535.60 | 1519.88 | 1517.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1525.80 | 1529.05 | 1523.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 1525.80 | 1529.05 | 1523.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1552.90 | 1533.82 | 1526.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 1526.70 | 1533.82 | 1526.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1555.90 | 1556.10 | 1549.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 1555.90 | 1556.10 | 1549.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1551.50 | 1555.56 | 1551.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 1551.50 | 1555.56 | 1551.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1556.30 | 1555.71 | 1551.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:30:00 | 1554.10 | 1555.71 | 1551.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1573.70 | 1560.78 | 1555.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 1596.70 | 1571.22 | 1562.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 1598.40 | 1580.06 | 1568.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 1626.70 | 1628.64 | 1628.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 1626.70 | 1628.64 | 1628.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 1626.70 | 1628.64 | 1628.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 1621.70 | 1627.25 | 1628.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 11:15:00 | 1568.30 | 1567.13 | 1582.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 11:45:00 | 1567.20 | 1567.13 | 1582.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1572.40 | 1559.64 | 1571.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 1541.90 | 1557.14 | 1565.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1572.00 | 1561.66 | 1561.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 12:15:00 | 1572.00 | 1561.66 | 1561.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 13:15:00 | 1576.20 | 1564.57 | 1562.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1562.20 | 1564.10 | 1562.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1562.20 | 1564.10 | 1562.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1559.60 | 1563.20 | 1562.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1545.50 | 1563.20 | 1562.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 1551.70 | 1560.90 | 1561.60 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 11:15:00 | 1566.40 | 1560.72 | 1559.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1585.80 | 1567.54 | 1563.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 10:15:00 | 1599.90 | 1600.93 | 1589.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1589.70 | 1597.05 | 1590.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1589.70 | 1597.05 | 1590.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1589.70 | 1597.05 | 1590.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1591.40 | 1595.92 | 1590.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1588.70 | 1595.92 | 1590.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1582.80 | 1593.30 | 1590.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1576.20 | 1593.30 | 1590.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1589.20 | 1592.48 | 1590.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 12:30:00 | 1592.50 | 1592.15 | 1590.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1595.20 | 1593.78 | 1591.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:30:00 | 1603.60 | 1595.51 | 1592.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:45:00 | 1593.90 | 1599.61 | 1597.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1594.90 | 1598.67 | 1597.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 1618.90 | 1602.71 | 1599.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 1657.80 | 1673.48 | 1674.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-21 13:15:00 | 1651.70 | 1666.16 | 1670.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1691.00 | 1668.53 | 1670.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 1691.00 | 1668.53 | 1670.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1686.90 | 1672.20 | 1672.39 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 1699.90 | 1677.74 | 1674.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 12:15:00 | 1709.90 | 1696.58 | 1690.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1702.00 | 1702.14 | 1695.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:00:00 | 1702.00 | 1702.14 | 1695.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1697.40 | 1702.18 | 1696.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 1699.60 | 1702.18 | 1696.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1694.00 | 1700.54 | 1696.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 1694.00 | 1700.54 | 1696.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1699.90 | 1700.41 | 1696.93 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1691.10 | 1695.84 | 1696.20 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 1705.90 | 1697.85 | 1697.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1714.00 | 1701.08 | 1698.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1699.30 | 1703.42 | 1700.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 1699.30 | 1703.42 | 1700.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 1697.00 | 1702.13 | 1700.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:30:00 | 1698.90 | 1702.13 | 1700.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 1703.00 | 1702.31 | 1700.63 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1688.20 | 1699.84 | 1699.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 12:15:00 | 1681.20 | 1694.30 | 1697.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1690.10 | 1690.04 | 1694.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1698.80 | 1690.04 | 1694.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1719.50 | 1695.93 | 1696.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1719.50 | 1695.93 | 1696.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 1724.80 | 1701.71 | 1699.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1751.50 | 1716.19 | 1706.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1747.60 | 1753.33 | 1738.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 1747.60 | 1753.33 | 1738.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1745.20 | 1752.16 | 1743.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1745.20 | 1752.16 | 1743.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1738.90 | 1749.51 | 1743.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 1744.60 | 1749.51 | 1743.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1744.90 | 1748.59 | 1743.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 1740.60 | 1748.59 | 1743.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1768.20 | 1752.51 | 1745.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:45:00 | 1770.10 | 1756.61 | 1748.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 14:15:00 | 1773.80 | 1762.62 | 1752.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1740.90 | 1753.30 | 1754.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 10:15:00 | 1740.90 | 1753.30 | 1754.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1740.90 | 1753.30 | 1754.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 11:15:00 | 1733.10 | 1749.26 | 1752.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 11:15:00 | 1740.10 | 1740.01 | 1745.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 12:00:00 | 1740.10 | 1740.01 | 1745.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1736.90 | 1735.99 | 1740.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:30:00 | 1743.30 | 1735.99 | 1740.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1735.00 | 1735.79 | 1740.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1735.00 | 1735.79 | 1740.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1733.50 | 1724.07 | 1730.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1733.50 | 1724.07 | 1730.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1734.00 | 1726.05 | 1730.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1729.00 | 1730.79 | 1732.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 1727.80 | 1731.85 | 1732.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1742.70 | 1733.10 | 1732.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 1742.70 | 1733.10 | 1732.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 1742.70 | 1733.10 | 1732.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1750.00 | 1739.67 | 1736.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1718.80 | 1736.94 | 1735.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1718.80 | 1736.94 | 1735.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1730.50 | 1735.65 | 1735.47 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 1729.90 | 1734.50 | 1734.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 12:15:00 | 1725.00 | 1732.60 | 1734.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1721.30 | 1721.07 | 1727.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 1721.30 | 1721.07 | 1727.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 1717.00 | 1710.71 | 1716.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 1717.00 | 1710.71 | 1716.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1715.20 | 1711.61 | 1716.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 09:15:00 | 1709.00 | 1711.61 | 1716.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 09:15:00 | 1723.30 | 1713.95 | 1716.80 | SL hit (close>static) qty=1.00 sl=1722.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:15:00 | 1713.10 | 1715.58 | 1717.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 1708.00 | 1712.79 | 1715.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 10:45:00 | 1710.10 | 1707.95 | 1711.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1715.60 | 1709.48 | 1711.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 1715.60 | 1709.48 | 1711.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1713.30 | 1710.25 | 1712.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 1691.00 | 1703.71 | 1708.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1724.60 | 1702.81 | 1707.16 | SL hit (close>static) qty=1.00 sl=1722.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1724.60 | 1702.81 | 1707.16 | SL hit (close>static) qty=1.00 sl=1722.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1724.60 | 1702.81 | 1707.16 | SL hit (close>static) qty=1.00 sl=1722.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1724.60 | 1702.81 | 1707.16 | SL hit (close>static) qty=1.00 sl=1716.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 1730.10 | 1713.34 | 1711.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1749.00 | 1727.36 | 1719.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1744.00 | 1745.50 | 1734.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1747.30 | 1745.50 | 1734.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1732.50 | 1742.43 | 1734.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1732.50 | 1742.43 | 1734.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1736.60 | 1741.27 | 1734.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 1737.20 | 1741.27 | 1734.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1738.50 | 1740.71 | 1735.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:30:00 | 1736.60 | 1740.71 | 1735.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1737.10 | 1740.42 | 1736.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:15:00 | 1736.00 | 1740.42 | 1736.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1746.20 | 1741.58 | 1737.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 10:30:00 | 1747.60 | 1741.40 | 1738.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 14:15:00 | 1731.80 | 1736.58 | 1737.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 14:15:00 | 1731.80 | 1736.58 | 1737.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1718.80 | 1728.86 | 1733.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1732.50 | 1728.53 | 1731.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1732.50 | 1728.53 | 1731.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1732.10 | 1729.25 | 1731.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 1726.00 | 1729.25 | 1731.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1720.00 | 1727.40 | 1730.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1715.00 | 1727.40 | 1730.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 13:15:00 | 1737.50 | 1729.00 | 1728.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 13:15:00 | 1737.50 | 1729.00 | 1728.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 1753.10 | 1736.21 | 1731.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1731.40 | 1736.30 | 1733.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1731.40 | 1736.30 | 1733.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1726.50 | 1734.34 | 1732.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1718.20 | 1734.34 | 1732.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1725.00 | 1732.47 | 1731.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1723.80 | 1732.47 | 1731.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1706.80 | 1727.34 | 1729.57 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 1751.60 | 1731.58 | 1728.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 1761.70 | 1742.02 | 1735.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1743.90 | 1747.13 | 1739.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1742.10 | 1747.13 | 1739.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1727.90 | 1743.28 | 1738.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1731.40 | 1743.28 | 1738.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1730.00 | 1740.62 | 1738.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:30:00 | 1735.90 | 1739.46 | 1737.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:00:00 | 1734.80 | 1739.46 | 1737.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 11:45:00 | 1738.40 | 1740.63 | 1738.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 14:15:00 | 1746.00 | 1738.35 | 1737.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1742.90 | 1739.26 | 1738.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:30:00 | 1761.90 | 1742.74 | 1740.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 11:00:00 | 1755.00 | 1745.20 | 1741.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 09:30:00 | 1754.80 | 1762.21 | 1752.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1836.60 | 1842.80 | 1843.64 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 1853.00 | 1845.40 | 1844.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 15:15:00 | 1855.00 | 1847.32 | 1845.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1846.60 | 1847.54 | 1846.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 1846.60 | 1847.54 | 1846.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 1839.20 | 1845.87 | 1845.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:45:00 | 1835.90 | 1845.87 | 1845.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 1847.60 | 1846.22 | 1845.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:00:00 | 1852.70 | 1847.52 | 1846.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1852.70 | 1847.90 | 1846.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:00:00 | 1852.80 | 1849.46 | 1847.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 1852.80 | 1850.47 | 1848.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1852.00 | 1851.97 | 1849.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 1851.30 | 1851.97 | 1849.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1855.70 | 1852.72 | 1850.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 1874.00 | 1857.90 | 1853.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:45:00 | 1872.70 | 1860.78 | 1854.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 14:45:00 | 1873.80 | 1865.16 | 1857.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1873.50 | 1865.15 | 1858.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1886.40 | 1869.40 | 1861.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:45:00 | 1896.50 | 1875.30 | 1864.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 12:00:00 | 1892.70 | 1878.78 | 1867.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1906.80 | 1927.23 | 1927.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1901.80 | 1922.15 | 1925.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 1911.80 | 1911.29 | 1917.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:00:00 | 1911.80 | 1911.29 | 1917.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 1902.20 | 1909.47 | 1915.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 1909.70 | 1909.47 | 1915.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1879.00 | 1903.13 | 1911.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 10:45:00 | 1869.00 | 1896.01 | 1907.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1874.00 | 1884.22 | 1891.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:30:00 | 1877.00 | 1872.91 | 1880.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1775.55 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1780.30 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 1783.15 | 1803.92 | 1827.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1787.00 | 1764.74 | 1788.47 | SL hit (close>ema200) qty=0.50 sl=1764.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1787.00 | 1764.74 | 1788.47 | SL hit (close>ema200) qty=0.50 sl=1764.74 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 1787.00 | 1764.74 | 1788.47 | SL hit (close>ema200) qty=0.50 sl=1764.74 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1684.40 | 1663.87 | 1661.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1728.50 | 1682.06 | 1671.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1706.90 | 1708.82 | 1693.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:00:00 | 1725.70 | 1712.20 | 1696.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:45:00 | 1720.80 | 1713.58 | 1704.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1743.90 | 1765.17 | 1767.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 1743.90 | 1765.17 | 1767.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 1743.90 | 1765.17 | 1767.04 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 15:15:00 | 1774.90 | 1759.18 | 1758.10 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 10:15:00 | 1749.90 | 1756.29 | 1756.90 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 1767.00 | 1759.11 | 1758.06 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 1744.40 | 1756.00 | 1757.17 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 1769.20 | 1757.31 | 1757.14 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 1747.00 | 1755.86 | 1756.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 1736.20 | 1751.92 | 1754.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1754.50 | 1751.04 | 1753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 1754.50 | 1751.04 | 1753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1753.80 | 1751.60 | 1753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 1755.90 | 1751.60 | 1753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1743.50 | 1749.98 | 1752.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 1725.90 | 1750.16 | 1752.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1739.90 | 1735.51 | 1741.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 15:15:00 | 1652.90 | 1677.33 | 1689.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1639.61 | 1670.91 | 1685.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 1620.00 | 1618.70 | 1639.07 | SL hit (close>ema200) qty=0.50 sl=1618.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 15:15:00 | 1620.00 | 1618.70 | 1639.07 | SL hit (close>ema200) qty=0.50 sl=1618.70 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1585.20 | 1558.35 | 1555.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1595.10 | 1565.70 | 1558.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1582.60 | 1599.70 | 1585.33 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 1570.10 | 1579.95 | 1581.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 1553.50 | 1574.66 | 1578.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1509.70 | 1493.70 | 1513.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1509.70 | 1493.70 | 1513.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1515.90 | 1498.14 | 1513.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 1515.70 | 1498.14 | 1513.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1502.50 | 1499.01 | 1512.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1510.30 | 1499.01 | 1512.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1571.70 | 1515.15 | 1517.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1571.70 | 1515.15 | 1517.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1577.30 | 1527.58 | 1523.10 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 1518.30 | 1530.38 | 1531.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 1502.20 | 1524.75 | 1529.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 1519.00 | 1512.61 | 1520.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 12:00:00 | 1519.00 | 1512.61 | 1520.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 1517.20 | 1513.53 | 1520.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 1517.20 | 1513.53 | 1520.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 1512.30 | 1513.28 | 1519.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:30:00 | 1515.70 | 1513.28 | 1519.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 1500.00 | 1509.89 | 1516.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 1517.20 | 1509.89 | 1516.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1523.10 | 1512.53 | 1517.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:30:00 | 1501.50 | 1513.20 | 1516.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1476.10 | 1514.74 | 1516.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1524.90 | 1514.64 | 1514.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1524.90 | 1514.64 | 1514.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1524.90 | 1514.64 | 1514.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1538.50 | 1522.80 | 1518.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1695.50 | 1702.18 | 1668.96 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1719.00 | 1702.18 | 1668.96 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1749.30 | 1751.59 | 1718.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 1761.50 | 1753.91 | 1727.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:00:00 | 1761.00 | 1755.33 | 1730.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1765.00 | 1752.46 | 1733.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:45:00 | 1763.50 | 1753.39 | 1735.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 09:15:00 | 1804.95 | 1771.16 | 1755.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 1765.00 | 1770.09 | 1757.87 | SL hit (close<ema200) qty=0.50 sl=1770.09 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 1774.60 | 1787.73 | 1779.25 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 14:15:00 | 1768.30 | 1775.17 | 1775.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 15:15:00 | 1766.60 | 1773.46 | 1774.63 | Break + close below crossover candle low |

### Cycle 65 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1808.70 | 1780.50 | 1777.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1832.60 | 1790.92 | 1782.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 09:15:00 | 1805.10 | 1806.17 | 1796.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 09:45:00 | 1803.10 | 1806.17 | 1796.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1793.20 | 1806.69 | 1801.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 1794.90 | 1806.69 | 1801.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1787.30 | 1802.81 | 1800.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:45:00 | 1788.80 | 1802.81 | 1800.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 1797.20 | 1800.16 | 1799.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:00:00 | 1797.20 | 1800.16 | 1799.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 1789.20 | 1797.97 | 1798.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 1780.00 | 1794.38 | 1797.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 14:15:00 | 1774.10 | 1770.98 | 1782.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 15:00:00 | 1774.10 | 1770.98 | 1782.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1776.60 | 1772.11 | 1781.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1787.90 | 1772.11 | 1781.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1786.50 | 1774.99 | 1782.12 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 1792.90 | 1786.80 | 1786.14 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1756.70 | 1782.27 | 1784.29 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 1818.50 | 1784.12 | 1781.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1834.10 | 1794.11 | 1786.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 1791.50 | 1795.65 | 1788.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 1791.50 | 1795.65 | 1788.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1793.00 | 1795.12 | 1788.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1733.40 | 1795.12 | 1788.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1739.60 | 1784.02 | 1784.41 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1790.80 | 1779.43 | 1778.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1812.40 | 1798.09 | 1791.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 1825.20 | 1825.60 | 1815.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 1824.80 | 1825.60 | 1815.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1824.70 | 1825.42 | 1816.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 1841.00 | 1826.80 | 1820.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 15:15:00 | 1845.00 | 1826.80 | 1820.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 09:30:00 | 1510.00 | 2025-05-12 13:15:00 | 1531.30 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-12 10:45:00 | 1513.10 | 2025-05-12 13:15:00 | 1531.30 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1546.30 | 2025-05-23 15:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.83% |
| BUY | retest2 | 2025-05-16 11:15:00 | 1549.80 | 2025-05-23 15:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-05-28 12:30:00 | 1604.50 | 2025-05-28 13:15:00 | 1596.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-07-08 09:15:00 | 1559.90 | 2025-07-09 09:15:00 | 1517.40 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1501.30 | 2025-07-14 14:15:00 | 1512.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-07-14 13:00:00 | 1501.50 | 2025-07-14 14:15:00 | 1512.20 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-07-15 09:30:00 | 1501.20 | 2025-07-17 10:15:00 | 1511.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-17 09:30:00 | 1499.90 | 2025-07-17 10:15:00 | 1511.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-07-22 12:30:00 | 1465.40 | 2025-07-25 09:15:00 | 1522.60 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-07-23 09:30:00 | 1461.00 | 2025-07-25 09:15:00 | 1522.60 | STOP_HIT | 1.00 | -4.22% |
| BUY | retest1 | 2025-07-28 10:45:00 | 1524.60 | 2025-07-29 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest1 | 2025-07-28 11:15:00 | 1524.10 | 2025-07-29 09:15:00 | 1492.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-07-29 15:15:00 | 1511.50 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-07-30 11:15:00 | 1513.90 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-30 11:45:00 | 1511.10 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-07-30 14:30:00 | 1512.20 | 2025-07-30 15:15:00 | 1499.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-06 09:15:00 | 1456.00 | 2025-08-13 09:15:00 | 1453.30 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2025-09-15 13:15:00 | 1596.70 | 2025-09-23 14:15:00 | 1626.70 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-09-16 09:15:00 | 1598.40 | 2025-09-23 14:15:00 | 1626.70 | STOP_HIT | 1.00 | 1.77% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1541.90 | 2025-10-01 12:15:00 | 1572.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-09 12:30:00 | 1592.50 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2025-10-09 15:00:00 | 1595.20 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 3.92% |
| BUY | retest2 | 2025-10-10 09:30:00 | 1603.60 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 3.38% |
| BUY | retest2 | 2025-10-13 09:45:00 | 1593.90 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 4.01% |
| BUY | retest2 | 2025-10-13 12:00:00 | 1618.90 | 2025-10-20 14:15:00 | 1657.80 | STOP_HIT | 1.00 | 2.40% |
| BUY | retest2 | 2025-11-07 11:45:00 | 1770.10 | 2025-11-11 10:15:00 | 1740.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-07 14:15:00 | 1773.80 | 2025-11-11 10:15:00 | 1740.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1729.00 | 2025-11-17 10:15:00 | 1742.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1727.80 | 2025-11-17 10:15:00 | 1742.70 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-21 09:15:00 | 1709.00 | 2025-11-21 09:15:00 | 1723.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-21 11:15:00 | 1713.10 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-11-21 12:45:00 | 1708.00 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-24 10:45:00 | 1710.10 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-11-24 14:45:00 | 1691.00 | 2025-11-25 09:15:00 | 1724.60 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-12-01 10:30:00 | 1747.60 | 2025-12-01 14:15:00 | 1731.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1715.00 | 2025-12-04 13:15:00 | 1737.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-11 10:30:00 | 1735.90 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.80% |
| BUY | retest2 | 2025-12-11 11:00:00 | 1734.80 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.87% |
| BUY | retest2 | 2025-12-11 11:45:00 | 1738.40 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.65% |
| BUY | retest2 | 2025-12-11 14:15:00 | 1746.00 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2025-12-12 09:30:00 | 1761.90 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.24% |
| BUY | retest2 | 2025-12-12 11:00:00 | 1755.00 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.65% |
| BUY | retest2 | 2025-12-15 09:30:00 | 1754.80 | 2025-12-29 12:15:00 | 1836.60 | STOP_HIT | 1.00 | 4.66% |
| BUY | retest2 | 2025-12-30 14:00:00 | 1852.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1852.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.92% |
| BUY | retest2 | 2025-12-31 12:00:00 | 1852.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2025-12-31 12:45:00 | 1852.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-01-01 11:45:00 | 1874.00 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2026-01-01 12:45:00 | 1872.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-01 14:45:00 | 1873.80 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2026-01-02 09:15:00 | 1873.50 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2026-01-02 10:45:00 | 1896.50 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2026-01-02 12:00:00 | 1892.70 | 2026-01-08 12:15:00 | 1906.80 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2026-01-12 10:45:00 | 1869.00 | 2026-01-20 15:15:00 | 1775.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1874.00 | 2026-01-20 15:15:00 | 1780.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 09:30:00 | 1877.00 | 2026-01-20 15:15:00 | 1783.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 10:45:00 | 1869.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.39% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1874.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2026-01-16 09:30:00 | 1877.00 | 2026-01-22 09:15:00 | 1787.00 | STOP_HIT | 0.50 | 4.79% |
| BUY | retest2 | 2026-02-05 11:00:00 | 1725.70 | 2026-02-13 10:15:00 | 1743.90 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2026-02-06 10:45:00 | 1720.80 | 2026-02-13 10:15:00 | 1743.90 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1725.90 | 2026-02-27 15:15:00 | 1652.90 | PARTIAL | 0.50 | 4.23% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1739.90 | 2026-03-02 09:15:00 | 1639.61 | PARTIAL | 0.50 | 5.76% |
| SELL | retest2 | 2026-02-20 09:15:00 | 1725.90 | 2026-03-04 15:15:00 | 1620.00 | STOP_HIT | 0.50 | 6.14% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1739.90 | 2026-03-04 15:15:00 | 1620.00 | STOP_HIT | 0.50 | 6.89% |
| SELL | retest2 | 2026-04-01 11:30:00 | 1501.50 | 2026-04-02 15:15:00 | 1524.90 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1476.10 | 2026-04-02 15:15:00 | 1524.90 | STOP_HIT | 1.00 | -3.31% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1719.00 | 2026-04-16 09:15:00 | 1804.95 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1719.00 | 2026-04-16 11:15:00 | 1765.00 | STOP_HIT | 0.50 | 2.68% |
| BUY | retest2 | 2026-04-13 13:00:00 | 1761.50 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2026-04-13 14:00:00 | 1761.00 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1765.00 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2026-04-15 09:45:00 | 1763.50 | 2026-04-20 14:15:00 | 1768.30 | STOP_HIT | 1.00 | 0.27% |
