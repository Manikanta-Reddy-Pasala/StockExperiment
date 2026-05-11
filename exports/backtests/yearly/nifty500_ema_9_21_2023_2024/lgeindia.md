# LG Electronics India Ltd. (LGEINDIA)

## Backtest Summary

- **Window:** 2025-10-14 09:15:00 → 2026-05-08 15:15:00 (968 bars)
- **Last close:** 1508.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 38 |
| ALERT1 | 28 |
| ALERT2 | 28 |
| ALERT2_SKIP | 14 |
| ALERT3 | 76 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 32 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 5 / 31
- **Target hits / Stop hits / Partials:** 1 / 34 / 1
- **Avg / median % per leg:** -1.13% / -1.27%
- **Sum % (uncompounded):** -40.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 3 | 14.3% | 1 | 20 | 0 | -1.42% | -29.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 3 | 14.3% | 1 | 20 | 0 | -1.42% | -29.8% |
| SELL (all) | 15 | 2 | 13.3% | 0 | 14 | 1 | -0.74% | -11.1% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.17% | -8.7% |
| SELL @ 3rd Alert (retest2) | 11 | 2 | 18.2% | 0 | 10 | 1 | -0.22% | -2.4% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.17% | -8.7% |
| retest2 (combined) | 32 | 5 | 15.6% | 1 | 30 | 1 | -1.01% | -32.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1677.90 | 1670.85 | 1670.63 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1660.30 | 1670.53 | 1670.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 1658.70 | 1668.17 | 1669.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 10:15:00 | 1650.70 | 1649.76 | 1656.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:30:00 | 1650.50 | 1649.76 | 1656.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1653.10 | 1650.70 | 1654.16 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 14:15:00 | 1658.30 | 1655.73 | 1655.50 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 1650.80 | 1654.74 | 1655.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1650.00 | 1653.79 | 1654.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 1655.10 | 1652.70 | 1653.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 14:15:00 | 1655.10 | 1652.70 | 1653.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 1655.10 | 1652.70 | 1653.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 1648.70 | 1652.97 | 1653.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 1664.70 | 1655.31 | 1654.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1664.70 | 1655.31 | 1654.59 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1640.50 | 1656.20 | 1657.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1630.60 | 1645.87 | 1650.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 15:15:00 | 1624.10 | 1618.44 | 1629.66 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1592.40 | 1618.44 | 1629.66 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-07 11:15:00 | 1612.80 | 1614.52 | 1625.72 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1610.50 | 1611.52 | 1619.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-10 12:15:00 | 1661.90 | 1621.29 | 1621.57 | SL hit (close>ema400) qty=1.00 sl=1621.57 alert=retest1 |

### Cycle 7 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 1654.20 | 1627.88 | 1624.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1682.60 | 1664.16 | 1656.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 1670.00 | 1670.71 | 1662.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 1635.00 | 1670.71 | 1662.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1632.40 | 1663.05 | 1659.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1640.40 | 1658.52 | 1658.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 1629.90 | 1652.80 | 1655.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1629.90 | 1652.80 | 1655.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1622.80 | 1646.80 | 1652.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 1631.30 | 1628.50 | 1637.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1625.40 | 1628.50 | 1637.06 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 15:15:00 | 1624.80 | 1628.06 | 1636.08 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1635.40 | 1629.26 | 1634.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-18 10:15:00 | 1635.40 | 1629.26 | 1634.59 | SL hit (close>ema400) qty=1.00 sl=1634.59 alert=retest1 |

### Cycle 9 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 1670.00 | 1640.92 | 1637.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 1681.00 | 1659.68 | 1647.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1668.90 | 1677.45 | 1667.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1668.90 | 1677.45 | 1667.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1668.90 | 1677.45 | 1667.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 1665.70 | 1677.45 | 1667.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1678.50 | 1677.66 | 1668.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:45:00 | 1665.60 | 1677.66 | 1668.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1653.60 | 1672.85 | 1667.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:45:00 | 1654.90 | 1672.85 | 1667.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 1656.20 | 1669.52 | 1666.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:30:00 | 1661.60 | 1667.38 | 1665.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 1633.00 | 1658.46 | 1661.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 1633.00 | 1658.46 | 1661.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1622.20 | 1648.01 | 1656.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1645.40 | 1632.65 | 1642.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1645.40 | 1632.65 | 1642.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1645.40 | 1632.65 | 1642.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:30:00 | 1639.80 | 1632.65 | 1642.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1634.80 | 1633.08 | 1642.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:30:00 | 1628.00 | 1632.42 | 1641.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:00:00 | 1629.80 | 1632.42 | 1641.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 1629.50 | 1632.20 | 1639.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 1625.00 | 1631.96 | 1638.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1627.60 | 1629.98 | 1636.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 10:15:00 | 1637.90 | 1635.22 | 1635.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 1637.90 | 1635.22 | 1635.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 13:15:00 | 1656.00 | 1642.01 | 1638.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 1636.50 | 1646.23 | 1641.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1636.50 | 1646.23 | 1641.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1636.50 | 1646.23 | 1641.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 1636.50 | 1646.23 | 1641.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1648.20 | 1646.62 | 1642.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:15:00 | 1652.00 | 1646.62 | 1642.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1650.00 | 1648.03 | 1643.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1633.50 | 1646.86 | 1644.87 | SL hit (close<static) qty=1.00 sl=1636.10 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1635.00 | 1642.59 | 1643.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1624.10 | 1637.68 | 1640.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 1617.50 | 1612.13 | 1620.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 1617.50 | 1612.13 | 1620.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1617.50 | 1612.13 | 1620.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1617.50 | 1612.13 | 1620.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1607.70 | 1608.94 | 1615.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 1613.00 | 1608.94 | 1615.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1592.40 | 1574.46 | 1584.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 1594.80 | 1574.46 | 1584.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 1591.90 | 1577.95 | 1585.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 1591.90 | 1577.95 | 1585.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1591.90 | 1588.23 | 1588.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 1591.90 | 1588.23 | 1588.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 1569.50 | 1567.41 | 1573.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 1569.50 | 1567.41 | 1573.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 1575.10 | 1569.06 | 1573.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:15:00 | 1561.20 | 1568.47 | 1572.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1556.40 | 1571.09 | 1571.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 1581.00 | 1551.98 | 1548.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1581.00 | 1551.98 | 1548.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 12:15:00 | 1581.70 | 1569.41 | 1559.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 1565.20 | 1571.09 | 1562.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 14:15:00 | 1565.20 | 1571.09 | 1562.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1565.20 | 1571.09 | 1562.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 1565.20 | 1571.09 | 1562.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1557.00 | 1568.28 | 1561.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 1570.90 | 1566.88 | 1562.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 1546.60 | 1558.98 | 1559.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1546.60 | 1558.98 | 1559.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1535.50 | 1554.28 | 1557.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1543.40 | 1530.49 | 1536.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1543.40 | 1530.49 | 1536.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1543.40 | 1530.49 | 1536.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 1555.40 | 1530.49 | 1536.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1552.00 | 1534.79 | 1538.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1553.70 | 1534.79 | 1538.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 13:15:00 | 1547.50 | 1540.25 | 1540.07 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 1526.00 | 1538.31 | 1539.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 1524.90 | 1535.63 | 1538.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 1482.30 | 1472.84 | 1483.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 1482.30 | 1472.84 | 1483.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1482.30 | 1472.84 | 1483.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 1482.30 | 1472.84 | 1483.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 1489.20 | 1476.11 | 1484.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 1489.20 | 1476.11 | 1484.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 1487.00 | 1478.29 | 1484.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:00:00 | 1487.00 | 1478.29 | 1484.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1488.90 | 1482.09 | 1485.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:15:00 | 1488.00 | 1482.09 | 1485.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1494.90 | 1484.65 | 1486.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 15:00:00 | 1494.90 | 1484.65 | 1486.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 1496.00 | 1486.92 | 1486.91 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 1485.00 | 1486.54 | 1486.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 1477.00 | 1484.63 | 1485.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1401.00 | 1394.47 | 1408.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:15:00 | 1413.30 | 1394.47 | 1408.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1409.50 | 1397.48 | 1408.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:45:00 | 1398.10 | 1403.44 | 1408.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1328.19 | 1356.81 | 1366.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 11:15:00 | 1357.70 | 1356.80 | 1364.64 | SL hit (close>ema200) qty=0.50 sl=1356.80 alert=retest2 |

### Cycle 19 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 1397.20 | 1372.40 | 1369.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 13:15:00 | 1403.40 | 1398.60 | 1394.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 1412.50 | 1412.97 | 1405.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 14:30:00 | 1412.70 | 1412.97 | 1405.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1427.90 | 1416.04 | 1408.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 1440.60 | 1422.01 | 1411.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 13:30:00 | 1430.80 | 1434.18 | 1428.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 15:15:00 | 1400.00 | 1422.45 | 1424.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1400.00 | 1422.45 | 1424.20 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 1434.70 | 1426.06 | 1425.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1492.70 | 1441.40 | 1432.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1503.80 | 1514.66 | 1494.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 1503.80 | 1514.66 | 1494.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1503.80 | 1514.66 | 1494.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 1497.30 | 1514.66 | 1494.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1501.00 | 1513.55 | 1504.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1501.00 | 1513.55 | 1504.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1506.80 | 1512.20 | 1504.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 1499.60 | 1512.20 | 1504.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1499.70 | 1508.71 | 1504.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 1499.70 | 1508.71 | 1504.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1505.70 | 1508.11 | 1504.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 1496.70 | 1508.11 | 1504.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 1502.80 | 1507.05 | 1504.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 15:00:00 | 1502.80 | 1507.05 | 1504.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1499.00 | 1505.44 | 1503.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1497.30 | 1503.81 | 1503.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1503.60 | 1503.77 | 1503.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 1498.30 | 1503.77 | 1503.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 1515.80 | 1506.17 | 1504.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 12:15:00 | 1517.50 | 1506.17 | 1504.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:45:00 | 1521.00 | 1513.39 | 1508.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 1525.60 | 1534.33 | 1526.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:00:00 | 1519.20 | 1529.00 | 1524.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1520.50 | 1526.10 | 1524.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:45:00 | 1518.60 | 1526.10 | 1524.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1450.00 | 1508.83 | 1516.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1450.00 | 1508.83 | 1516.68 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 1528.60 | 1503.20 | 1503.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 11:15:00 | 1549.00 | 1520.97 | 1514.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 12:15:00 | 1552.00 | 1555.76 | 1546.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 12:45:00 | 1551.90 | 1555.76 | 1546.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1549.80 | 1553.64 | 1546.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:45:00 | 1549.00 | 1553.64 | 1546.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1542.30 | 1551.37 | 1546.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1537.90 | 1551.37 | 1546.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1552.00 | 1551.50 | 1546.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 1554.90 | 1551.50 | 1546.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:00:00 | 1554.00 | 1553.04 | 1548.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:15:00 | 1568.00 | 1551.43 | 1549.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 14:00:00 | 1561.30 | 1561.45 | 1559.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1558.00 | 1563.02 | 1561.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 1560.00 | 1563.02 | 1561.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 1558.00 | 1562.01 | 1561.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 1555.10 | 1562.01 | 1561.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1562.10 | 1562.03 | 1561.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 1555.60 | 1560.42 | 1560.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 1555.60 | 1560.42 | 1560.66 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 09:15:00 | 1577.50 | 1563.84 | 1562.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 13:15:00 | 1591.30 | 1575.23 | 1568.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 1573.70 | 1578.93 | 1572.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 1573.70 | 1578.93 | 1572.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1573.70 | 1578.93 | 1572.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 1592.60 | 1579.95 | 1574.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 11:15:00 | 1550.00 | 1569.51 | 1571.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2026-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 11:15:00 | 1550.00 | 1569.51 | 1571.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 12:15:00 | 1544.80 | 1564.57 | 1569.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1565.00 | 1557.43 | 1563.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1565.00 | 1557.43 | 1563.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1565.00 | 1557.43 | 1563.57 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 1578.50 | 1568.17 | 1567.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1600.00 | 1578.05 | 1572.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 1580.80 | 1585.41 | 1579.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 1580.80 | 1585.41 | 1579.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1580.80 | 1585.41 | 1579.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1580.80 | 1585.41 | 1579.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1581.00 | 1584.53 | 1579.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 1527.00 | 1584.53 | 1579.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1523.60 | 1572.34 | 1574.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 1521.00 | 1562.07 | 1569.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 1556.60 | 1544.50 | 1555.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 1556.60 | 1544.50 | 1555.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1556.60 | 1544.50 | 1555.05 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1585.00 | 1562.24 | 1561.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 1592.00 | 1572.28 | 1566.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1570.50 | 1571.93 | 1567.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:00:00 | 1570.50 | 1571.93 | 1567.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1574.00 | 1572.34 | 1567.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:30:00 | 1577.00 | 1573.01 | 1568.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 13:00:00 | 1575.70 | 1573.01 | 1568.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 1553.30 | 1573.86 | 1570.92 | SL hit (close<static) qty=1.00 sl=1566.10 alert=retest2 |

### Cycle 30 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 1552.60 | 1570.97 | 1571.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 1546.20 | 1566.02 | 1568.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 1559.90 | 1558.89 | 1564.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 14:00:00 | 1559.90 | 1558.89 | 1564.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 1561.50 | 1559.41 | 1564.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:45:00 | 1571.00 | 1559.41 | 1564.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 1551.10 | 1557.75 | 1563.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:30:00 | 1565.60 | 1557.68 | 1562.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 1542.10 | 1550.24 | 1556.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 1551.70 | 1550.24 | 1556.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1551.50 | 1547.22 | 1553.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1558.90 | 1547.22 | 1553.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1563.70 | 1550.52 | 1554.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 1563.70 | 1550.52 | 1554.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 11:15:00 | 1578.90 | 1556.19 | 1557.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:00:00 | 1578.90 | 1556.19 | 1557.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1575.50 | 1560.05 | 1558.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 1590.30 | 1568.55 | 1563.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1547.80 | 1572.33 | 1569.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1547.80 | 1572.33 | 1569.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1547.80 | 1572.33 | 1569.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 1547.80 | 1572.33 | 1569.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1560.80 | 1566.79 | 1567.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 1538.60 | 1559.34 | 1563.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1552.00 | 1549.87 | 1557.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 1552.00 | 1549.87 | 1557.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1558.00 | 1551.50 | 1557.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1558.00 | 1551.50 | 1557.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1562.80 | 1553.76 | 1557.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 12:00:00 | 1562.80 | 1553.76 | 1557.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1553.20 | 1553.65 | 1557.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1516.50 | 1555.91 | 1557.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 1538.80 | 1506.73 | 1505.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 14:15:00 | 1538.80 | 1506.73 | 1505.94 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1473.20 | 1500.87 | 1503.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 1450.20 | 1490.73 | 1498.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 10:15:00 | 1343.30 | 1340.42 | 1378.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-06 10:45:00 | 1343.30 | 1340.42 | 1378.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1379.70 | 1354.62 | 1376.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:15:00 | 1388.00 | 1354.62 | 1376.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1380.40 | 1359.78 | 1376.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 1383.80 | 1359.78 | 1376.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1383.10 | 1367.61 | 1377.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:30:00 | 1383.70 | 1367.61 | 1377.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 1372.60 | 1368.61 | 1376.89 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1426.70 | 1388.79 | 1384.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 11:15:00 | 1444.20 | 1407.86 | 1394.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 1422.00 | 1422.21 | 1407.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 1422.00 | 1422.21 | 1407.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1463.70 | 1460.04 | 1443.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1479.00 | 1460.04 | 1443.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1626.90 | 1497.92 | 1473.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 1551.50 | 1583.64 | 1586.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1532.60 | 1553.73 | 1568.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 13:15:00 | 1547.20 | 1545.82 | 1560.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 14:00:00 | 1547.20 | 1545.82 | 1560.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1580.00 | 1552.66 | 1562.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1580.00 | 1552.66 | 1562.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1580.00 | 1558.12 | 1563.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 1597.30 | 1558.12 | 1563.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1624.30 | 1576.80 | 1571.75 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 13:15:00 | 1589.40 | 1605.94 | 1606.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 1583.40 | 1597.14 | 1601.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 10:15:00 | 1566.70 | 1562.35 | 1573.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:00:00 | 1566.70 | 1562.35 | 1573.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1575.70 | 1565.15 | 1573.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1575.70 | 1565.15 | 1573.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1585.20 | 1569.16 | 1574.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:00:00 | 1585.20 | 1569.16 | 1574.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1567.20 | 1568.77 | 1573.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 1557.10 | 1565.93 | 1571.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-30 09:45:00 | 1648.70 | 2025-10-30 10:15:00 | 1664.70 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest1 | 2025-11-07 09:15:00 | 1592.40 | 2025-11-10 12:15:00 | 1661.90 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest1 | 2025-11-07 11:15:00 | 1612.80 | 2025-11-10 12:15:00 | 1661.90 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2025-11-14 11:00:00 | 1640.40 | 2025-11-14 11:15:00 | 1629.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2025-11-17 14:15:00 | 1625.40 | 2025-11-18 10:15:00 | 1635.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2025-11-17 15:15:00 | 1624.80 | 2025-11-18 10:15:00 | 1635.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-11-18 15:00:00 | 1623.20 | 2025-11-19 09:15:00 | 1665.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-11-21 13:30:00 | 1661.60 | 2025-11-21 15:15:00 | 1633.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-11-25 11:30:00 | 1628.00 | 2025-11-28 10:15:00 | 1637.90 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-11-25 12:00:00 | 1629.80 | 2025-11-28 10:15:00 | 1637.90 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-25 14:15:00 | 1629.50 | 2025-11-28 10:15:00 | 1637.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1625.00 | 2025-11-28 10:15:00 | 1637.90 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-12-01 11:15:00 | 1652.00 | 2025-12-02 09:15:00 | 1633.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1650.00 | 2025-12-02 09:15:00 | 1633.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-12-12 11:15:00 | 1561.20 | 2025-12-19 14:15:00 | 1581.00 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1556.40 | 2025-12-19 14:15:00 | 1581.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-12-23 11:15:00 | 1570.90 | 2025-12-23 15:15:00 | 1546.60 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-13 14:45:00 | 1398.10 | 2026-01-21 09:15:00 | 1328.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:45:00 | 1398.10 | 2026-01-21 11:15:00 | 1357.70 | STOP_HIT | 0.50 | 2.89% |
| BUY | retest2 | 2026-01-30 10:30:00 | 1440.60 | 2026-02-01 15:15:00 | 1400.00 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-02-01 13:30:00 | 1430.80 | 2026-02-01 15:15:00 | 1400.00 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2026-02-09 12:15:00 | 1517.50 | 2026-02-12 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest2 | 2026-02-09 14:45:00 | 1521.00 | 2026-02-12 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -4.67% |
| BUY | retest2 | 2026-02-11 10:15:00 | 1525.60 | 2026-02-12 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest2 | 2026-02-11 12:00:00 | 1519.20 | 2026-02-12 09:15:00 | 1450.00 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-02-20 10:15:00 | 1554.90 | 2026-02-26 15:15:00 | 1555.60 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2026-02-20 12:00:00 | 1554.00 | 2026-02-26 15:15:00 | 1555.60 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2026-02-23 10:15:00 | 1568.00 | 2026-02-26 15:15:00 | 1555.60 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-02-25 14:00:00 | 1561.30 | 2026-02-26 15:15:00 | 1555.60 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2026-03-02 14:45:00 | 1592.60 | 2026-03-04 11:15:00 | 1550.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-03-11 12:30:00 | 1577.00 | 2026-03-12 09:15:00 | 1553.30 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-03-11 13:00:00 | 1575.70 | 2026-03-12 09:15:00 | 1553.30 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2026-03-12 12:00:00 | 1579.00 | 2026-03-13 09:15:00 | 1552.60 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2026-03-12 12:30:00 | 1582.00 | 2026-03-13 09:15:00 | 1552.60 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1516.50 | 2026-03-27 14:15:00 | 1538.80 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1479.00 | 2026-04-15 09:15:00 | 1626.90 | TARGET_HIT | 1.00 | 10.00% |
