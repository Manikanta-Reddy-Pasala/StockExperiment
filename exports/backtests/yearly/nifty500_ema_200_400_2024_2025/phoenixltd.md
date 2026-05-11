# Phoenix Mills Ltd. (PHOENIXLTD)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 5 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 51 |
| PARTIAL | 13 |
| TARGET_HIT | 13 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 35
- **Target hits / Stop hits / Partials:** 13 / 42 / 13
- **Avg / median % per leg:** 1.59% / -0.36%
- **Sum % (uncompounded):** 108.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 7 | 35.0% | 7 | 13 | 0 | 2.01% | 40.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.06% | -3.1% |
| BUY @ 3rd Alert (retest2) | 19 | 7 | 36.8% | 7 | 12 | 0 | 2.28% | 43.3% |
| SELL (all) | 48 | 26 | 54.2% | 6 | 29 | 13 | 1.42% | 68.0% |
| SELL @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 3.38% | 20.3% |
| SELL @ 3rd Alert (retest2) | 42 | 20 | 47.6% | 6 | 26 | 10 | 1.14% | 47.7% |
| retest1 (combined) | 7 | 6 | 85.7% | 0 | 4 | 3 | 2.46% | 17.2% |
| retest2 (combined) | 61 | 27 | 44.3% | 13 | 38 | 10 | 1.49% | 91.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-08 12:15:00 | 1670.40 | 1751.45 | 1751.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 13:15:00 | 1652.40 | 1750.46 | 1751.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1749.65 | 1748.58 | 1750.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1749.65 | 1748.58 | 1750.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1749.65 | 1748.58 | 1750.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1749.65 | 1748.58 | 1750.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1736.65 | 1748.46 | 1750.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 1750.40 | 1748.46 | 1750.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1602.00 | 1542.46 | 1605.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:30:00 | 1603.50 | 1542.46 | 1605.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 1630.45 | 1543.33 | 1605.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 15:00:00 | 1630.45 | 1543.33 | 1605.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 15:15:00 | 1616.00 | 1544.05 | 1605.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:15:00 | 1656.30 | 1544.05 | 1605.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1791.00 | 1643.98 | 1643.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1819.00 | 1647.27 | 1645.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 1706.80 | 1707.82 | 1680.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 12:00:00 | 1722.65 | 1707.99 | 1680.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 1682.05 | 1707.64 | 1680.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 1682.05 | 1707.64 | 1680.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1669.95 | 1707.26 | 1680.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-19 14:15:00 | 1669.95 | 1707.26 | 1680.76 | SL hit (close<ema400) qty=1.00 sl=1680.76 alert=retest1 |

### Cycle 3 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 1550.00 | 1666.49 | 1666.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1518.55 | 1654.76 | 1660.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 12:15:00 | 1657.60 | 1644.88 | 1654.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 12:15:00 | 1657.60 | 1644.88 | 1654.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 1657.60 | 1644.88 | 1654.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 1657.60 | 1644.88 | 1654.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 1655.50 | 1644.99 | 1654.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:30:00 | 1668.40 | 1644.99 | 1654.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 1658.35 | 1645.12 | 1654.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 14:30:00 | 1658.55 | 1645.12 | 1654.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 1648.40 | 1645.16 | 1654.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 1679.40 | 1645.51 | 1654.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1666.45 | 1645.72 | 1654.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 13:45:00 | 1660.50 | 1646.40 | 1654.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 15:00:00 | 1654.95 | 1646.49 | 1654.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 12:30:00 | 1659.70 | 1647.11 | 1655.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 1660.45 | 1647.11 | 1655.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1643.45 | 1647.15 | 1655.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 14:30:00 | 1655.00 | 1647.15 | 1655.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1627.00 | 1646.87 | 1654.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 13:30:00 | 1614.40 | 1645.81 | 1654.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 1612.00 | 1645.81 | 1654.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 15:00:00 | 1609.60 | 1645.45 | 1653.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 1577.47 | 1644.22 | 1653.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 1572.20 | 1644.22 | 1653.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 1576.71 | 1644.22 | 1653.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 1577.43 | 1644.22 | 1653.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 14:15:00 | 1533.68 | 1639.39 | 1650.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 14:15:00 | 1531.40 | 1639.39 | 1650.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 14:15:00 | 1529.12 | 1639.39 | 1650.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 09:15:00 | 1494.45 | 1636.82 | 1649.11 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 1649.30 | 1610.78 | 1610.68 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1544.75 | 1610.43 | 1610.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-11 13:15:00 | 1474.30 | 1592.44 | 1601.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 14:15:00 | 1593.70 | 1586.10 | 1597.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 14:15:00 | 1593.70 | 1586.10 | 1597.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 1593.70 | 1586.10 | 1597.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:45:00 | 1607.50 | 1586.10 | 1597.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 1596.00 | 1586.20 | 1597.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 1598.50 | 1586.20 | 1597.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1600.40 | 1586.34 | 1597.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:15:00 | 1614.70 | 1586.34 | 1597.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 1613.60 | 1586.61 | 1597.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:30:00 | 1624.00 | 1586.61 | 1597.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 1626.20 | 1588.57 | 1597.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:45:00 | 1626.00 | 1588.57 | 1597.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 09:15:00 | 1662.30 | 1605.87 | 1605.87 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 13:15:00 | 1535.00 | 1606.02 | 1606.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 1519.50 | 1602.37 | 1604.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 1588.40 | 1567.34 | 1583.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 09:15:00 | 1588.40 | 1567.34 | 1583.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1588.40 | 1567.34 | 1583.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:00:00 | 1588.40 | 1567.34 | 1583.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 1612.00 | 1567.78 | 1583.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 1612.00 | 1567.78 | 1583.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 1607.00 | 1568.17 | 1583.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:30:00 | 1606.40 | 1568.17 | 1583.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1596.40 | 1570.10 | 1584.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:30:00 | 1600.10 | 1570.10 | 1584.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1588.10 | 1570.87 | 1584.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:15:00 | 1621.80 | 1570.87 | 1584.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1610.60 | 1571.26 | 1584.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 1587.40 | 1575.47 | 1585.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 1582.80 | 1575.47 | 1585.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1587.30 | 1577.85 | 1586.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 1584.50 | 1581.36 | 1587.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1578.00 | 1581.36 | 1587.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1564.50 | 1581.33 | 1587.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 1564.00 | 1578.82 | 1585.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 1602.10 | 1579.22 | 1585.58 | SL hit (close>static) qty=1.00 sl=1599.30 alert=retest2 |

### Cycle 8 — BUY (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 13:15:00 | 1644.70 | 1590.34 | 1590.23 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 1538.00 | 1592.38 | 1592.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 11:15:00 | 1499.50 | 1591.45 | 1591.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 10:15:00 | 1528.00 | 1524.00 | 1550.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 1508.00 | 1524.39 | 1549.57 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 13:45:00 | 1513.30 | 1524.02 | 1548.76 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-28 14:45:00 | 1511.20 | 1523.88 | 1548.57 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:15:00 | 1437.63 | 1507.18 | 1534.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-06 10:15:00 | 1435.64 | 1507.18 | 1534.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:15:00 | 1432.60 | 1503.44 | 1531.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-18 10:15:00 | 1484.10 | 1481.15 | 1513.94 | SL hit (close>ema200) qty=0.50 sl=1481.15 alert=retest1 |

### Cycle 10 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1602.60 | 1528.51 | 1528.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1616.10 | 1532.31 | 1530.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1558.40 | 1563.55 | 1548.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-25 15:00:00 | 1558.40 | 1563.55 | 1548.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1555.00 | 1563.39 | 1548.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 11:45:00 | 1567.20 | 1563.41 | 1548.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 12:45:00 | 1567.00 | 1563.41 | 1548.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 1579.60 | 1562.90 | 1548.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:45:00 | 1570.40 | 1562.99 | 1548.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1555.70 | 1562.99 | 1549.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:30:00 | 1550.30 | 1562.99 | 1549.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1549.60 | 1562.85 | 1549.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1549.60 | 1562.85 | 1549.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1551.00 | 1562.73 | 1549.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1541.90 | 1562.73 | 1549.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1547.40 | 1562.58 | 1549.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:30:00 | 1562.50 | 1562.11 | 1549.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 09:30:00 | 1565.90 | 1562.04 | 1549.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:00:00 | 1562.50 | 1562.05 | 1549.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 11:45:00 | 1563.10 | 1562.10 | 1549.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1551.70 | 1562.21 | 1549.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 1542.20 | 1562.01 | 1549.79 | SL hit (close<static) qty=1.00 sl=1544.50 alert=retest2 |

### Cycle 11 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 1748.70 | 1758.36 | 1758.37 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 13:15:00 | 1769.20 | 1758.45 | 1758.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 1800.20 | 1758.97 | 1758.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1733.00 | 1761.35 | 1759.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 1733.00 | 1761.35 | 1759.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1743.90 | 1761.17 | 1759.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 1748.20 | 1761.06 | 1759.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:00:00 | 1748.90 | 1760.94 | 1759.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 13:30:00 | 1744.70 | 1760.76 | 1759.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:45:00 | 1746.80 | 1760.07 | 1759.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1752.70 | 1759.94 | 1759.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 1720.40 | 1758.39 | 1758.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1720.40 | 1758.39 | 1758.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 1720.10 | 1756.03 | 1757.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 1599.30 | 1587.42 | 1641.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 1599.30 | 1587.42 | 1641.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1709.20 | 1588.76 | 1641.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 1710.90 | 1588.76 | 1641.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 1812.40 | 1679.02 | 1678.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 1834.10 | 1709.11 | 1694.94 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-12-19 12:00:00 | 1722.65 | 2024-12-19 14:15:00 | 1669.95 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-12-24 09:15:00 | 1681.45 | 2024-12-31 09:15:00 | 1632.50 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2024-12-26 12:30:00 | 1674.15 | 2024-12-31 09:15:00 | 1632.50 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-12-26 13:30:00 | 1677.10 | 2024-12-31 09:15:00 | 1632.50 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-12-26 14:45:00 | 1722.40 | 2024-12-31 09:15:00 | 1632.50 | STOP_HIT | 1.00 | -5.22% |
| SELL | retest2 | 2025-01-16 13:45:00 | 1660.50 | 2025-01-21 09:15:00 | 1577.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 15:00:00 | 1654.95 | 2025-01-21 09:15:00 | 1572.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 12:30:00 | 1659.70 | 2025-01-21 09:15:00 | 1576.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-17 13:15:00 | 1660.45 | 2025-01-21 09:15:00 | 1577.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 13:30:00 | 1614.40 | 2025-01-21 14:15:00 | 1533.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 14:15:00 | 1612.00 | 2025-01-21 14:15:00 | 1531.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-20 15:00:00 | 1609.60 | 2025-01-21 14:15:00 | 1529.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 13:45:00 | 1660.50 | 2025-01-22 09:15:00 | 1494.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-16 15:00:00 | 1654.95 | 2025-01-22 09:15:00 | 1489.46 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-17 12:30:00 | 1659.70 | 2025-01-22 09:15:00 | 1493.73 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-17 13:15:00 | 1660.45 | 2025-01-22 09:15:00 | 1494.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 13:30:00 | 1614.40 | 2025-01-22 12:15:00 | 1452.96 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 14:15:00 | 1612.00 | 2025-01-22 13:15:00 | 1450.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-20 15:00:00 | 1609.60 | 2025-01-30 09:15:00 | 1600.00 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2025-01-31 09:30:00 | 1609.50 | 2025-02-01 12:15:00 | 1720.00 | STOP_HIT | 1.00 | -6.87% |
| SELL | retest2 | 2025-02-06 14:15:00 | 1603.95 | 2025-02-07 10:15:00 | 1658.05 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2025-02-10 10:00:00 | 1603.50 | 2025-02-17 09:15:00 | 1523.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 11:45:00 | 1606.90 | 2025-02-17 09:15:00 | 1526.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 13:15:00 | 1610.50 | 2025-02-17 09:15:00 | 1529.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:00:00 | 1603.50 | 2025-03-05 14:15:00 | 1581.25 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-02-10 11:45:00 | 1606.90 | 2025-03-05 14:15:00 | 1581.25 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2025-02-10 13:15:00 | 1610.50 | 2025-03-05 14:15:00 | 1581.25 | STOP_HIT | 0.50 | 1.82% |
| SELL | retest2 | 2025-03-10 11:45:00 | 1602.00 | 2025-03-11 13:15:00 | 1644.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-03-10 12:15:00 | 1591.95 | 2025-03-11 13:15:00 | 1644.00 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2025-03-12 12:30:00 | 1600.65 | 2025-03-18 11:15:00 | 1606.35 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-03-12 15:00:00 | 1600.00 | 2025-03-18 11:15:00 | 1606.35 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-03-13 10:45:00 | 1569.65 | 2025-03-20 11:15:00 | 1606.50 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-13 11:15:00 | 1571.45 | 2025-03-21 09:15:00 | 1682.75 | STOP_HIT | 1.00 | -7.08% |
| SELL | retest2 | 2025-03-19 09:45:00 | 1569.50 | 2025-03-21 09:15:00 | 1682.75 | STOP_HIT | 1.00 | -7.22% |
| SELL | retest2 | 2025-05-22 13:45:00 | 1587.40 | 2025-06-03 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-05-22 14:15:00 | 1582.80 | 2025-06-03 13:15:00 | 1602.10 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1587.30 | 2025-06-06 15:15:00 | 1609.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-28 15:15:00 | 1584.50 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2025-05-30 09:15:00 | 1564.50 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.21% |
| SELL | retest2 | 2025-06-03 11:00:00 | 1564.00 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.24% |
| SELL | retest2 | 2025-06-05 14:45:00 | 1565.10 | 2025-06-10 09:15:00 | 1646.00 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest1 | 2025-07-28 09:15:00 | 1508.00 | 2025-08-06 10:15:00 | 1437.63 | PARTIAL | 0.50 | 4.67% |
| SELL | retest1 | 2025-07-28 13:45:00 | 1513.30 | 2025-08-06 10:15:00 | 1435.64 | PARTIAL | 0.50 | 5.13% |
| SELL | retest1 | 2025-07-28 14:45:00 | 1511.20 | 2025-08-07 09:15:00 | 1432.60 | PARTIAL | 0.50 | 5.20% |
| SELL | retest1 | 2025-07-28 09:15:00 | 1508.00 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest1 | 2025-07-28 13:45:00 | 1513.30 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.93% |
| SELL | retest1 | 2025-07-28 14:45:00 | 1511.20 | 2025-08-18 10:15:00 | 1484.10 | STOP_HIT | 0.50 | 1.79% |
| SELL | retest2 | 2025-08-28 12:45:00 | 1518.90 | 2025-09-02 09:15:00 | 1566.90 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-09-03 12:45:00 | 1517.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-04 10:00:00 | 1515.00 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-08 09:45:00 | 1516.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-08 11:45:00 | 1523.00 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1511.30 | 2025-09-09 10:15:00 | 1540.80 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-26 11:45:00 | 1567.20 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-26 12:45:00 | 1567.00 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-09-29 09:15:00 | 1579.60 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-09-29 09:45:00 | 1570.40 | 2025-10-03 10:15:00 | 1542.20 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-09-30 14:30:00 | 1562.50 | 2025-10-28 09:15:00 | 1718.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 09:30:00 | 1565.90 | 2025-10-28 09:15:00 | 1722.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 11:00:00 | 1562.50 | 2025-10-28 09:15:00 | 1718.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-01 11:45:00 | 1563.10 | 2025-10-28 09:15:00 | 1719.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 14:30:00 | 1573.10 | 2025-11-03 09:15:00 | 1723.70 | TARGET_HIT | 1.00 | 9.57% |
| BUY | retest2 | 2025-10-06 11:15:00 | 1567.00 | 2025-11-03 10:15:00 | 1730.41 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2025-10-06 12:45:00 | 1570.90 | 2025-11-03 10:15:00 | 1727.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 11:30:00 | 1748.20 | 2026-02-20 09:15:00 | 1720.40 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-02-13 13:00:00 | 1748.90 | 2026-02-20 09:15:00 | 1720.40 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-02-13 13:30:00 | 1744.70 | 2026-02-20 09:15:00 | 1720.40 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-02-16 09:45:00 | 1746.80 | 2026-02-20 09:15:00 | 1720.40 | STOP_HIT | 1.00 | -1.51% |
