# Prestige Estates Projects Ltd. (PRESTIGE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1495.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 52 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 19 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 20 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 20
- **Target hits / Stop hits / Partials:** 1 / 20 / 1
- **Avg / median % per leg:** -1.32% / -1.94%
- **Sum % (uncompounded):** -29.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -2.20% | -44.0% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.50% | -7.0% |
| BUY @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -2.06% | -37.0% |
| SELL (all) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -3.50% | -7.0% |
| retest2 (combined) | 20 | 2 | 10.0% | 1 | 18 | 1 | -1.10% | -22.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 11:15:00 | 1389.40 | 1284.38 | 1284.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1394.40 | 1287.55 | 1285.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 10:15:00 | 1610.50 | 1612.98 | 1519.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 11:00:00 | 1610.50 | 1612.98 | 1519.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1642.00 | 1683.90 | 1606.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 1623.70 | 1683.90 | 1606.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1606.10 | 1681.98 | 1606.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 1606.10 | 1681.98 | 1606.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 1598.30 | 1681.15 | 1606.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:45:00 | 1597.30 | 1681.15 | 1606.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 1613.80 | 1680.48 | 1606.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 1622.90 | 1679.78 | 1606.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:45:00 | 1624.30 | 1669.66 | 1607.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 1631.50 | 1669.23 | 1608.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1626.10 | 1663.97 | 1608.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1608.30 | 1662.05 | 1608.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1608.30 | 1662.05 | 1608.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1618.40 | 1661.62 | 1609.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 1622.00 | 1661.62 | 1609.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1609.00 | 1660.68 | 1609.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 1609.00 | 1660.68 | 1609.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1602.20 | 1660.10 | 1609.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1602.20 | 1660.10 | 1609.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 14:15:00 | 1609.80 | 1659.60 | 1609.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1572.80 | 1658.22 | 1608.87 | SL hit (close<static) qty=1.00 sl=1591.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1572.80 | 1658.22 | 1608.87 | SL hit (close<static) qty=1.00 sl=1591.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1572.80 | 1658.22 | 1608.87 | SL hit (close<static) qty=1.00 sl=1591.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 1572.80 | 1658.22 | 1608.87 | SL hit (close<static) qty=1.00 sl=1591.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 13:00:00 | 1616.70 | 1656.20 | 1608.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1594.30 | 1654.07 | 1608.45 | SL hit (close<static) qty=1.00 sl=1598.20 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 1620.00 | 1654.07 | 1608.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:00:00 | 1616.90 | 1653.19 | 1608.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 1618.40 | 1652.37 | 1608.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1600.30 | 1651.36 | 1608.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 1593.40 | 1651.36 | 1608.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1610.10 | 1650.95 | 1608.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:30:00 | 1598.50 | 1650.95 | 1608.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1610.60 | 1650.55 | 1608.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 1610.60 | 1650.55 | 1608.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 1607.00 | 1650.12 | 1608.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 1607.00 | 1650.12 | 1608.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 1607.00 | 1649.69 | 1608.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 1607.00 | 1649.69 | 1608.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1600.60 | 1649.20 | 1608.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1600.60 | 1649.20 | 1608.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1607.00 | 1648.78 | 1608.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 09:15:00 | 1612.40 | 1648.78 | 1608.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:15:00 | 1617.00 | 1648.37 | 1608.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 1611.90 | 1645.64 | 1611.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 1627.40 | 1643.22 | 1610.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1616.30 | 1642.05 | 1611.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 1619.30 | 1642.05 | 1611.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1625.20 | 1640.93 | 1615.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1627.40 | 1640.93 | 1615.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1619.30 | 1640.49 | 1615.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:30:00 | 1617.30 | 1640.49 | 1615.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 1620.60 | 1640.29 | 1615.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 1620.00 | 1640.29 | 1615.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 1613.00 | 1640.02 | 1615.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:00:00 | 1613.00 | 1640.02 | 1615.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1608.20 | 1639.71 | 1615.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:45:00 | 1608.10 | 1639.71 | 1615.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1606.00 | 1639.37 | 1615.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1572.20 | 1639.37 | 1615.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1598.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1598.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1598.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1585.00 | 1638.83 | 1615.61 | SL hit (close<static) qty=1.00 sl=1596.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1594.80 | 1619.22 | 1608.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1594.80 | 1619.22 | 1608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1578.00 | 1618.81 | 1608.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 12:00:00 | 1578.00 | 1618.81 | 1608.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1555.00 | 1599.25 | 1599.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 13:15:00 | 1551.60 | 1596.32 | 1597.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 1593.60 | 1591.55 | 1595.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:00:00 | 1593.60 | 1591.55 | 1595.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1592.10 | 1591.55 | 1595.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1592.30 | 1591.55 | 1595.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1592.30 | 1591.56 | 1595.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 1592.30 | 1591.56 | 1595.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1603.20 | 1591.68 | 1595.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1596.60 | 1591.68 | 1595.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1611.70 | 1591.87 | 1595.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1611.70 | 1591.87 | 1595.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1608.90 | 1592.04 | 1595.52 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 11:15:00 | 1638.80 | 1598.77 | 1598.73 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 1522.90 | 1598.72 | 1598.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 12:15:00 | 1521.10 | 1595.25 | 1597.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1569.40 | 1566.11 | 1580.18 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1710.00 | 1591.53 | 1591.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 11:15:00 | 1720.20 | 1594.03 | 1592.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 1695.80 | 1696.71 | 1658.40 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1756.10 | 1696.71 | 1658.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-20 10:15:00 | 1707.40 | 1708.78 | 1671.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1676.60 | 1707.46 | 1673.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1673.80 | 1707.46 | 1673.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | SL hit (close<ema400) qty=1.00 sl=1673.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-11-24 10:15:00 | 1670.80 | 1707.09 | 1673.24 | SL hit (close<ema400) qty=1.00 sl=1673.24 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-11-24 10:45:00 | 1668.60 | 1707.09 | 1673.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 1660.10 | 1706.62 | 1673.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:00:00 | 1660.10 | 1706.62 | 1673.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1655.70 | 1701.83 | 1672.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1655.70 | 1701.83 | 1672.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1650.00 | 1701.32 | 1672.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:15:00 | 1643.80 | 1701.32 | 1672.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1659.00 | 1700.56 | 1672.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 11:00:00 | 1659.00 | 1700.56 | 1672.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1674.50 | 1699.72 | 1672.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:30:00 | 1672.50 | 1699.72 | 1672.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 1675.80 | 1699.48 | 1672.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 1675.80 | 1699.48 | 1672.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 1676.00 | 1699.25 | 1672.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:30:00 | 1665.70 | 1699.03 | 1672.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1673.30 | 1698.77 | 1672.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:30:00 | 1670.90 | 1698.77 | 1672.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1670.30 | 1698.49 | 1672.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 1670.30 | 1698.49 | 1672.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 1671.90 | 1698.22 | 1672.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 1668.20 | 1698.22 | 1672.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1670.40 | 1697.95 | 1672.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:30:00 | 1672.50 | 1696.53 | 1672.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 1672.60 | 1696.06 | 1672.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1651.50 | 1694.91 | 1672.12 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1651.50 | 1694.91 | 1672.12 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1673.00 | 1685.34 | 1669.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 1640.50 | 1685.31 | 1670.32 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1679.30 | 1672.59 | 1665.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1648.80 | 1672.34 | 1665.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 1648.80 | 1672.34 | 1665.31 | SL hit (close<static) qty=1.00 sl=1666.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 1645.20 | 1672.34 | 1665.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1645.90 | 1672.07 | 1665.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 13:15:00 | 1654.60 | 1671.80 | 1665.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1641.80 | 1671.12 | 1664.90 | SL hit (close<static) qty=1.00 sl=1643.70 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 14:30:00 | 1651.90 | 1669.71 | 1664.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1630.00 | 1669.12 | 1664.10 | SL hit (close<static) qty=1.00 sl=1643.70 alert=retest2 |

### Cycle 6 — SELL (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 14:15:00 | 1598.80 | 1659.09 | 1659.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1588.90 | 1642.07 | 1649.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 1638.00 | 1630.95 | 1642.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:45:00 | 1629.00 | 1630.95 | 1642.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 1643.30 | 1631.08 | 1642.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 1643.30 | 1631.08 | 1642.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 1641.80 | 1631.18 | 1642.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:30:00 | 1646.30 | 1631.18 | 1642.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 1643.80 | 1631.31 | 1642.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 1641.60 | 1631.31 | 1642.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1665.10 | 1631.65 | 1642.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 1665.10 | 1631.65 | 1642.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1635.80 | 1634.40 | 1643.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 1632.10 | 1634.40 | 1643.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1550.49 | 1630.53 | 1641.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-20 09:15:00 | 1468.89 | 1590.22 | 1617.01 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 09:15:00 | 1622.90 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.09% |
| BUY | retest2 | 2025-07-31 13:45:00 | 1624.30 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-07-31 14:30:00 | 1631.50 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1626.10 | 2025-08-06 09:15:00 | 1572.80 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-08-06 13:00:00 | 1616.70 | 2025-08-07 09:15:00 | 1594.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-08-07 09:30:00 | 1620.00 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-08-07 12:00:00 | 1616.90 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-07 14:00:00 | 1618.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1612.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-08-11 10:15:00 | 1617.00 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-08-14 10:00:00 | 1611.90 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1627.40 | 2025-08-28 09:15:00 | 1585.00 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest1 | 2025-11-13 09:15:00 | 1756.10 | 2025-11-24 10:15:00 | 1670.80 | STOP_HIT | 1.00 | -4.86% |
| BUY | retest1 | 2025-11-20 10:15:00 | 1707.40 | 2025-11-24 10:15:00 | 1670.80 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-28 11:30:00 | 1672.50 | 2025-12-01 11:15:00 | 1651.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-11-28 13:30:00 | 1672.60 | 2025-12-01 11:15:00 | 1651.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-12-05 09:15:00 | 1673.00 | 2025-12-08 09:15:00 | 1640.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1679.30 | 2025-12-12 10:15:00 | 1648.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-12-12 13:15:00 | 1654.60 | 2025-12-15 09:15:00 | 1641.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-15 14:30:00 | 1651.90 | 2025-12-16 09:15:00 | 1630.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1632.10 | 2026-01-09 09:15:00 | 1550.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 1632.10 | 2026-01-20 09:15:00 | 1468.89 | TARGET_HIT | 0.50 | 10.00% |
