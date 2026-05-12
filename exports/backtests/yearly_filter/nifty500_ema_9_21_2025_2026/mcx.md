# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3098.00
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
| ALERT2_SKIP | 36 |
| ALERT3 | 155 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 59 |
| PARTIAL | 1 |
| TARGET_HIT | 4 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 50
- **Target hits / Stop hits / Partials:** 4 / 55 / 1
- **Avg / median % per leg:** -0.26% / -0.96%
- **Sum % (uncompounded):** -15.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 8 | 25.0% | 4 | 28 | 0 | 0.75% | 23.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 8 | 25.0% | 4 | 28 | 0 | 0.75% | 23.9% |
| SELL (all) | 28 | 2 | 7.1% | 0 | 27 | 1 | -1.40% | -39.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 2 | 7.1% | 0 | 27 | 1 | -1.40% | -39.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 60 | 10 | 16.7% | 4 | 55 | 1 | -0.26% | -15.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1217.60 | 1194.46 | 1191.96 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 13:15:00 | 1188.40 | 1193.02 | 1193.14 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1207.20 | 1193.93 | 1193.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 1216.90 | 1198.52 | 1195.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1289.10 | 1291.43 | 1275.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 1282.40 | 1291.43 | 1275.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1272.20 | 1286.19 | 1278.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1272.20 | 1286.19 | 1278.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1267.00 | 1282.35 | 1277.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 1267.00 | 1282.35 | 1277.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 1279.40 | 1279.37 | 1276.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 11:00:00 | 1279.40 | 1279.37 | 1276.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 1264.00 | 1276.30 | 1275.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:00:00 | 1264.00 | 1276.30 | 1275.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 1274.40 | 1275.92 | 1275.58 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 1272.10 | 1275.15 | 1275.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 1251.40 | 1269.17 | 1272.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1277.60 | 1270.86 | 1272.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:00:00 | 1277.60 | 1270.86 | 1272.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 1274.00 | 1271.49 | 1272.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:15:00 | 1279.00 | 1271.49 | 1272.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 1272.80 | 1271.75 | 1272.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:30:00 | 1276.70 | 1271.75 | 1272.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 1273.10 | 1272.02 | 1272.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 14:45:00 | 1275.80 | 1272.02 | 1272.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 1273.60 | 1272.34 | 1273.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 1268.20 | 1272.34 | 1273.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1272.20 | 1272.31 | 1272.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 1272.20 | 1272.31 | 1272.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1278.60 | 1273.57 | 1273.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1294.60 | 1277.77 | 1275.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1288.80 | 1290.82 | 1284.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 1288.80 | 1290.82 | 1284.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1288.90 | 1290.43 | 1284.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:45:00 | 1288.10 | 1290.43 | 1284.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1294.40 | 1291.23 | 1285.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 13:15:00 | 1294.60 | 1291.23 | 1285.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1282.30 | 1289.53 | 1286.75 | SL hit (close<static) qty=1.00 sl=1285.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 1519.80 | 1534.03 | 1534.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1509.90 | 1527.83 | 1531.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 1518.10 | 1514.03 | 1519.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 1546.60 | 1514.03 | 1519.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1539.60 | 1519.15 | 1521.35 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 1542.20 | 1523.76 | 1523.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 14:15:00 | 1565.60 | 1544.73 | 1534.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1587.90 | 1588.95 | 1575.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:15:00 | 1584.40 | 1588.95 | 1575.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1569.10 | 1584.25 | 1575.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1569.10 | 1584.25 | 1575.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1570.70 | 1581.54 | 1575.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1559.70 | 1581.54 | 1575.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1559.70 | 1575.50 | 1573.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1559.70 | 1575.50 | 1573.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1554.90 | 1571.38 | 1571.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 1550.50 | 1564.47 | 1568.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1586.60 | 1566.45 | 1568.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1586.60 | 1566.45 | 1568.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 1592.50 | 1571.66 | 1570.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 12:15:00 | 1605.00 | 1578.33 | 1573.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1625.80 | 1638.15 | 1617.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1616.60 | 1638.15 | 1617.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1789.00 | 1801.28 | 1792.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 1789.00 | 1801.28 | 1792.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1786.40 | 1798.30 | 1791.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1782.20 | 1798.30 | 1791.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1811.80 | 1797.82 | 1792.67 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1777.40 | 1793.69 | 1793.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 15:15:00 | 1773.00 | 1781.38 | 1786.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1654.90 | 1636.59 | 1658.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 10:00:00 | 1654.90 | 1636.59 | 1658.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1644.00 | 1638.07 | 1657.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 1641.00 | 1639.38 | 1656.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:00:00 | 1638.60 | 1639.88 | 1653.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 1641.00 | 1640.11 | 1652.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 1667.00 | 1651.43 | 1653.59 | SL hit (close>static) qty=1.00 sl=1662.80 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1664.40 | 1655.94 | 1655.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1666.30 | 1659.43 | 1657.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1659.50 | 1659.82 | 1657.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:45:00 | 1657.20 | 1659.82 | 1657.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1658.00 | 1666.52 | 1663.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1658.00 | 1666.52 | 1663.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1646.00 | 1662.42 | 1661.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1646.00 | 1662.42 | 1661.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 1649.60 | 1659.85 | 1660.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 1634.00 | 1651.97 | 1656.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 1650.10 | 1647.90 | 1652.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:30:00 | 1648.10 | 1647.90 | 1652.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1637.50 | 1644.75 | 1650.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1654.70 | 1644.75 | 1650.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1651.00 | 1646.20 | 1649.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1651.00 | 1646.20 | 1649.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1659.40 | 1648.84 | 1650.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:45:00 | 1659.00 | 1648.84 | 1650.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 13:15:00 | 1648.00 | 1648.67 | 1650.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 09:30:00 | 1643.50 | 1649.62 | 1650.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1640.60 | 1647.70 | 1649.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 14:15:00 | 1643.90 | 1638.38 | 1641.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 15:15:00 | 1652.00 | 1643.30 | 1643.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 15:15:00 | 1652.00 | 1643.30 | 1643.12 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 1631.90 | 1641.02 | 1642.10 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 12:15:00 | 1649.40 | 1643.74 | 1643.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 13:15:00 | 1653.30 | 1645.65 | 1644.04 | Break + close above crossover candle high |

### Cycle 16 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1622.30 | 1641.81 | 1642.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1614.30 | 1636.31 | 1640.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 1566.30 | 1556.64 | 1578.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 1566.30 | 1556.64 | 1578.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1561.20 | 1561.56 | 1574.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1558.00 | 1561.56 | 1574.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 12:00:00 | 1560.60 | 1561.86 | 1572.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:30:00 | 1560.30 | 1562.30 | 1570.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 1558.70 | 1555.30 | 1561.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1595.70 | 1543.70 | 1547.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 1595.70 | 1543.70 | 1547.35 | SL hit (close>static) qty=1.00 sl=1581.70 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 1599.60 | 1554.88 | 1552.10 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1552.20 | 1567.23 | 1567.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 1547.60 | 1560.70 | 1564.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1558.60 | 1557.77 | 1562.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:30:00 | 1554.60 | 1557.77 | 1562.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1554.30 | 1557.07 | 1561.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:30:00 | 1543.80 | 1556.47 | 1560.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 14:15:00 | 1576.00 | 1560.38 | 1561.48 | SL hit (close>static) qty=1.00 sl=1563.10 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 1578.00 | 1563.90 | 1562.98 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1551.90 | 1560.59 | 1561.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1545.20 | 1556.02 | 1559.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1555.60 | 1552.30 | 1556.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 1559.20 | 1552.30 | 1556.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1566.40 | 1555.12 | 1557.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1566.40 | 1555.12 | 1557.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 1580.00 | 1560.10 | 1559.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1590.60 | 1570.17 | 1564.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1632.90 | 1652.40 | 1631.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:00:00 | 1632.90 | 1652.40 | 1631.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1622.60 | 1646.44 | 1630.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 1622.60 | 1646.44 | 1630.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1626.20 | 1642.39 | 1630.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:15:00 | 1630.80 | 1642.39 | 1630.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 1632.00 | 1636.77 | 1630.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 1643.40 | 1648.90 | 1648.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 1643.40 | 1648.90 | 1648.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1633.10 | 1644.70 | 1646.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1612.60 | 1602.58 | 1612.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1612.60 | 1602.58 | 1612.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1614.00 | 1604.86 | 1612.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 1613.00 | 1604.86 | 1612.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1611.00 | 1606.09 | 1612.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1603.00 | 1606.43 | 1612.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-28 09:15:00 | 1522.85 | 1547.85 | 1572.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1538.00 | 1494.74 | 1511.66 | SL hit (close>ema200) qty=0.50 sl=1494.74 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 12:15:00 | 1556.30 | 1523.33 | 1521.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 1563.60 | 1540.23 | 1530.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1536.40 | 1539.46 | 1531.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1533.70 | 1539.46 | 1531.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1536.50 | 1541.99 | 1535.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1536.50 | 1541.99 | 1535.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1530.50 | 1539.69 | 1534.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:45:00 | 1528.00 | 1539.69 | 1534.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1529.30 | 1537.61 | 1534.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1511.20 | 1537.61 | 1534.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1532.20 | 1535.36 | 1533.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:30:00 | 1523.10 | 1535.36 | 1533.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1541.80 | 1536.65 | 1534.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:45:00 | 1532.30 | 1536.65 | 1534.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1531.80 | 1535.68 | 1534.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1531.80 | 1535.68 | 1534.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1534.00 | 1535.34 | 1534.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:30:00 | 1532.60 | 1535.34 | 1534.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1541.40 | 1536.55 | 1534.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 1544.70 | 1537.20 | 1535.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 14:15:00 | 1530.20 | 1535.35 | 1535.30 | SL hit (close<static) qty=1.00 sl=1533.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 1521.60 | 1532.60 | 1534.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1518.40 | 1529.27 | 1532.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1529.60 | 1526.08 | 1529.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1525.10 | 1526.08 | 1529.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1528.40 | 1526.54 | 1528.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 1528.40 | 1526.54 | 1528.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 1536.70 | 1528.57 | 1529.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:00:00 | 1536.70 | 1528.57 | 1529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 1546.00 | 1532.06 | 1531.17 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 15:15:00 | 1524.20 | 1530.20 | 1530.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 1502.90 | 1524.74 | 1528.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1521.00 | 1519.06 | 1523.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 1519.30 | 1519.06 | 1523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 1524.00 | 1520.52 | 1523.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 1544.30 | 1520.52 | 1523.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1549.80 | 1526.38 | 1525.98 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 15:15:00 | 1524.00 | 1526.50 | 1526.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 1511.50 | 1523.50 | 1525.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 1524.10 | 1521.63 | 1523.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 1525.50 | 1521.63 | 1523.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 1520.80 | 1521.47 | 1523.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 14:30:00 | 1514.60 | 1519.87 | 1522.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1545.00 | 1523.99 | 1523.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 1545.00 | 1523.99 | 1523.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 1555.50 | 1535.90 | 1529.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1545.10 | 1549.36 | 1543.72 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 1523.70 | 1539.79 | 1540.38 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1576.00 | 1545.01 | 1541.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1594.80 | 1554.97 | 1546.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1603.30 | 1603.98 | 1592.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 1615.80 | 1606.56 | 1594.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 13:00:00 | 1610.20 | 1608.29 | 1597.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 09:15:00 | 1583.40 | 1599.45 | 1596.50 | SL hit (close<static) qty=1.00 sl=1590.10 alert=retest2 |

### Cycle 32 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 1589.30 | 1598.03 | 1598.31 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 1604.90 | 1599.41 | 1598.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 13:15:00 | 1616.00 | 1604.53 | 1601.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1604.40 | 1604.52 | 1602.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1608.60 | 1604.52 | 1602.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-26 09:45:00 | 1617.00 | 1604.53 | 1602.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 11:15:00 | 1597.80 | 1602.36 | 1601.69 | SL hit (close<static) qty=1.00 sl=1598.70 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 1596.00 | 1601.09 | 1601.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 1581.40 | 1597.15 | 1599.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1605.40 | 1595.02 | 1597.50 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 1600.10 | 1599.14 | 1599.01 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 1587.80 | 1596.87 | 1597.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 09:15:00 | 1568.20 | 1590.04 | 1594.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1586.10 | 1567.53 | 1575.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 1586.10 | 1567.53 | 1575.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1582.30 | 1570.48 | 1576.48 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 1599.50 | 1580.88 | 1580.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1638.70 | 1595.76 | 1587.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1640.80 | 1645.58 | 1634.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 1635.20 | 1645.58 | 1634.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1649.60 | 1645.88 | 1636.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:45:00 | 1652.40 | 1642.94 | 1639.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:30:00 | 1659.90 | 1647.73 | 1641.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-14 09:15:00 | 1817.64 | 1793.58 | 1756.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 11:15:00 | 1834.00 | 1861.67 | 1864.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 12:15:00 | 1832.20 | 1855.78 | 1861.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1849.00 | 1844.12 | 1851.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:45:00 | 1845.40 | 1844.12 | 1851.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1854.10 | 1846.12 | 1852.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:15:00 | 1865.00 | 1846.12 | 1852.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1857.20 | 1848.33 | 1852.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:30:00 | 1862.40 | 1848.33 | 1852.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1850.40 | 1851.41 | 1853.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 1855.70 | 1851.41 | 1853.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1848.50 | 1850.83 | 1852.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 1841.30 | 1850.83 | 1852.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 13:15:00 | 1859.00 | 1839.19 | 1839.53 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 14:15:00 | 1859.80 | 1843.31 | 1841.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 15:15:00 | 1864.20 | 1847.49 | 1843.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1841.60 | 1846.31 | 1843.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1845.30 | 1846.31 | 1843.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1823.40 | 1841.73 | 1841.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 1823.40 | 1841.73 | 1841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1823.70 | 1838.12 | 1839.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1812.70 | 1827.95 | 1833.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 1829.80 | 1825.80 | 1831.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 11:45:00 | 1830.40 | 1825.80 | 1831.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1824.30 | 1825.50 | 1831.07 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1845.30 | 1829.20 | 1829.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 1852.50 | 1833.86 | 1831.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 10:15:00 | 1894.70 | 1895.42 | 1876.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 10:45:00 | 1888.10 | 1895.42 | 1876.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 1872.70 | 1888.02 | 1877.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 13:45:00 | 1875.00 | 1888.02 | 1877.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1864.70 | 1883.35 | 1876.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 1864.70 | 1883.35 | 1876.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1865.60 | 1879.80 | 1875.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1862.90 | 1879.80 | 1875.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1848.00 | 1869.78 | 1871.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1839.40 | 1863.70 | 1868.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1863.10 | 1855.95 | 1861.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:45:00 | 1847.10 | 1855.95 | 1861.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 1866.60 | 1858.08 | 1862.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 10:45:00 | 1873.60 | 1858.08 | 1862.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1868.60 | 1860.19 | 1862.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:15:00 | 1868.80 | 1860.19 | 1862.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1873.40 | 1862.83 | 1863.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 1881.00 | 1862.83 | 1863.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1891.80 | 1868.62 | 1866.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1904.60 | 1880.86 | 1873.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 13:15:00 | 1890.80 | 1890.82 | 1881.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:45:00 | 1885.30 | 1890.82 | 1881.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 1910.80 | 1903.85 | 1892.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1912.00 | 1899.32 | 1893.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:45:00 | 1916.00 | 1900.40 | 1894.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 1877.20 | 1894.48 | 1892.71 | SL hit (close<static) qty=1.00 sl=1892.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 14:15:00 | 1887.20 | 1891.56 | 1891.73 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1932.30 | 1898.82 | 1894.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1940.80 | 1911.76 | 1901.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 13:15:00 | 1926.00 | 1926.42 | 1916.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 14:00:00 | 1926.00 | 1926.42 | 1916.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1929.60 | 1937.98 | 1931.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 1951.60 | 1940.94 | 1936.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:15:00 | 1950.50 | 1942.21 | 1937.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:00:00 | 1954.00 | 1960.27 | 1957.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 1950.00 | 1956.97 | 1956.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 1936.40 | 1952.86 | 1954.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 1936.40 | 1952.86 | 1954.91 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 1971.00 | 1957.98 | 1956.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 1983.20 | 1972.28 | 1965.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1971.30 | 1973.23 | 1967.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1971.30 | 1973.23 | 1967.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 2027.60 | 2059.57 | 2039.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 2027.60 | 2059.57 | 2039.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 2022.00 | 2052.06 | 2037.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:45:00 | 2025.20 | 2052.06 | 2037.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1998.40 | 2028.96 | 2029.43 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 2047.20 | 2029.01 | 2028.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 2061.80 | 2043.16 | 2036.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 2045.60 | 2047.73 | 2042.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 14:45:00 | 2038.80 | 2047.73 | 2042.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 2045.00 | 2047.19 | 2042.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 2055.60 | 2047.19 | 2042.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2039.80 | 2045.71 | 2042.19 | SL hit (close<static) qty=1.00 sl=2041.40 alert=retest2 |

### Cycle 50 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 2018.80 | 2040.18 | 2042.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 2005.60 | 2033.26 | 2038.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 2037.00 | 2027.05 | 2032.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:15:00 | 2038.20 | 2027.05 | 2032.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 2046.40 | 2030.92 | 2033.85 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 2051.40 | 2036.98 | 2036.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 13:15:00 | 2062.20 | 2042.03 | 2038.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 2053.60 | 2058.17 | 2049.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 2053.60 | 2058.17 | 2049.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2046.80 | 2055.90 | 2049.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:30:00 | 2048.20 | 2055.90 | 2049.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 2039.80 | 2052.68 | 2048.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 2037.80 | 2052.68 | 2048.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 2024.20 | 2042.52 | 2044.35 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 2068.80 | 2047.63 | 2046.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 2074.00 | 2057.56 | 2051.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2054.40 | 2058.69 | 2053.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:45:00 | 2051.00 | 2058.69 | 2053.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2039.80 | 2054.92 | 2052.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 2039.80 | 2054.92 | 2052.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 2031.20 | 2050.17 | 2050.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:00:00 | 2031.20 | 2050.17 | 2050.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 2018.60 | 2043.86 | 2047.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 2014.40 | 2037.97 | 2044.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 2006.20 | 2002.84 | 2021.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 12:00:00 | 2006.20 | 2002.84 | 2021.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2019.60 | 1998.94 | 2011.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 2018.60 | 1998.94 | 2011.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 2009.60 | 2001.07 | 2011.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:15:00 | 2024.00 | 2001.07 | 2011.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2033.80 | 2007.62 | 2013.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 2033.80 | 2007.62 | 2013.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 2027.80 | 2011.65 | 2014.51 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 2032.60 | 2019.03 | 2017.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 11:15:00 | 2051.40 | 2030.61 | 2023.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2012.80 | 2030.06 | 2026.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 2012.80 | 2030.06 | 2026.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2036.80 | 2031.41 | 2027.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 2012.00 | 2031.41 | 2027.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 2029.00 | 2030.93 | 2027.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:00:00 | 2029.00 | 2030.93 | 2027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 2030.20 | 2030.78 | 2027.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:15:00 | 2041.80 | 2030.26 | 2028.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 2039.80 | 2030.56 | 2028.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 2020.20 | 2027.76 | 2027.75 | SL hit (close<static) qty=1.00 sl=2024.20 alert=retest2 |

### Cycle 56 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 2011.00 | 2024.41 | 2026.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 14:15:00 | 2005.20 | 2018.77 | 2023.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 2015.40 | 2013.77 | 2019.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 11:00:00 | 2015.40 | 2013.77 | 2019.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2030.00 | 2017.01 | 2020.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2029.80 | 2017.01 | 2020.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2044.60 | 2022.53 | 2022.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 2044.60 | 2022.53 | 2022.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 2030.80 | 2024.18 | 2023.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 09:15:00 | 2060.20 | 2034.45 | 2028.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 2165.60 | 2167.44 | 2147.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 14:30:00 | 2162.80 | 2167.44 | 2147.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2189.00 | 2205.06 | 2190.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 2196.20 | 2205.06 | 2190.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2189.00 | 2201.85 | 2190.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2177.40 | 2201.85 | 2190.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2187.00 | 2198.88 | 2189.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 2204.00 | 2198.88 | 2189.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 2219.40 | 2187.36 | 2186.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:45:00 | 2207.40 | 2216.43 | 2206.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 15:15:00 | 2203.00 | 2204.53 | 2203.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2245.00 | 2212.38 | 2207.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-05 13:15:00 | 2188.00 | 2207.42 | 2208.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2188.00 | 2207.42 | 2208.93 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 2257.00 | 2214.62 | 2211.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 2298.00 | 2250.49 | 2233.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2278.00 | 2289.11 | 2266.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 2278.00 | 2289.11 | 2266.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2267.00 | 2282.43 | 2267.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:30:00 | 2263.00 | 2282.43 | 2267.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 2243.00 | 2274.55 | 2265.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 2243.00 | 2274.55 | 2265.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2230.00 | 2265.64 | 2262.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 15:00:00 | 2230.00 | 2265.64 | 2262.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 2231.00 | 2258.71 | 2259.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 2221.00 | 2244.05 | 2251.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2227.00 | 2215.11 | 2231.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 2227.00 | 2215.11 | 2231.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 2212.00 | 2214.49 | 2230.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 2197.00 | 2214.49 | 2230.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 2291.00 | 2239.02 | 2235.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 2291.00 | 2239.02 | 2235.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 09:15:00 | 2305.00 | 2279.09 | 2261.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 13:15:00 | 2424.00 | 2448.95 | 2415.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:00:00 | 2424.00 | 2448.95 | 2415.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 2395.00 | 2432.06 | 2415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 2383.00 | 2432.06 | 2415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 2391.00 | 2423.85 | 2413.52 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 2372.00 | 2407.42 | 2407.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 2349.00 | 2395.74 | 2402.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 13:15:00 | 2310.00 | 2309.48 | 2338.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 13:45:00 | 2316.00 | 2309.48 | 2338.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 2323.00 | 2314.86 | 2333.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 2312.00 | 2318.95 | 2332.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:30:00 | 2313.00 | 2316.36 | 2330.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 2390.00 | 2321.51 | 2326.97 | SL hit (close>static) qty=1.00 sl=2355.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 10:15:00 | 2371.00 | 2331.41 | 2330.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 2440.00 | 2381.76 | 2358.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 2617.00 | 2647.25 | 2576.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 2601.00 | 2647.25 | 2576.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 2611.00 | 2640.00 | 2579.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 2594.00 | 2640.00 | 2579.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 2597.00 | 2631.40 | 2581.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 2589.00 | 2631.40 | 2581.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 2589.00 | 2618.22 | 2583.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:15:00 | 2555.00 | 2618.22 | 2583.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 2546.00 | 2603.77 | 2580.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 2545.00 | 2603.77 | 2580.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 2515.00 | 2586.02 | 2574.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 2275.20 | 2586.02 | 2574.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 2229.60 | 2514.73 | 2543.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2127.30 | 2387.05 | 2476.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 2253.50 | 2249.08 | 2333.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 14:00:00 | 2253.50 | 2249.08 | 2333.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2340.00 | 2278.59 | 2332.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2485.30 | 2278.59 | 2332.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2412.30 | 2305.33 | 2340.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2411.40 | 2305.33 | 2340.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 11:15:00 | 2408.30 | 2326.89 | 2346.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:00:00 | 2402.40 | 2341.99 | 2351.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2427.80 | 2369.62 | 2363.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 2427.80 | 2369.62 | 2363.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 2484.30 | 2412.48 | 2386.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2434.00 | 2492.61 | 2450.99 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 2350.10 | 2419.11 | 2428.17 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 2437.00 | 2414.94 | 2413.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 2472.80 | 2433.63 | 2423.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 2396.00 | 2446.85 | 2438.60 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 2367.50 | 2430.98 | 2432.13 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 15:15:00 | 2438.30 | 2414.25 | 2411.88 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2343.60 | 2400.12 | 2405.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 2278.00 | 2341.57 | 2369.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 2341.30 | 2330.04 | 2351.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 2341.30 | 2330.04 | 2351.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 2350.00 | 2334.03 | 2351.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 09:15:00 | 2311.20 | 2334.03 | 2351.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 2337.50 | 2322.71 | 2322.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 2337.50 | 2322.71 | 2322.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 2401.10 | 2338.39 | 2329.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 2359.50 | 2370.59 | 2352.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 15:00:00 | 2359.50 | 2370.59 | 2352.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 2365.50 | 2369.57 | 2353.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 2398.00 | 2369.57 | 2353.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 2525.00 | 2558.04 | 2559.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 2525.00 | 2558.04 | 2559.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 2481.80 | 2540.71 | 2551.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 2534.00 | 2526.80 | 2539.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 2532.60 | 2526.80 | 2539.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2496.30 | 2520.41 | 2533.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 2479.70 | 2520.41 | 2533.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 11:00:00 | 2480.90 | 2512.51 | 2528.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 12:30:00 | 2483.10 | 2501.96 | 2521.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:30:00 | 2485.80 | 2499.57 | 2518.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 2494.10 | 2498.74 | 2513.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 2507.70 | 2498.74 | 2513.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 2513.80 | 2499.49 | 2510.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 12:00:00 | 2513.80 | 2499.49 | 2510.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 12:15:00 | 2515.00 | 2502.59 | 2511.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:00:00 | 2515.00 | 2502.59 | 2511.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 13:15:00 | 2538.10 | 2509.69 | 2513.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 13:45:00 | 2536.20 | 2509.69 | 2513.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-16 14:15:00 | 2549.50 | 2517.66 | 2517.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-03-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-16 14:15:00 | 2549.50 | 2517.66 | 2517.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 09:15:00 | 2623.60 | 2545.28 | 2530.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 2609.90 | 2649.95 | 2612.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 2609.90 | 2649.95 | 2612.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 2634.00 | 2646.76 | 2614.12 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 2561.00 | 2595.73 | 2599.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 2546.30 | 2585.84 | 2595.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 2330.00 | 2328.05 | 2385.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 2336.20 | 2328.05 | 2385.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2392.00 | 2346.99 | 2384.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 2392.00 | 2346.99 | 2384.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 2419.80 | 2361.55 | 2387.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 2419.80 | 2361.55 | 2387.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 2483.00 | 2410.92 | 2405.77 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 2398.60 | 2423.28 | 2424.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 2361.50 | 2407.20 | 2416.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 12:15:00 | 2410.00 | 2396.48 | 2408.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:00:00 | 2410.00 | 2396.48 | 2408.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 2403.00 | 2397.79 | 2407.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 13:30:00 | 2401.00 | 2397.79 | 2407.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 2390.50 | 2396.33 | 2406.32 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 2457.20 | 2417.11 | 2413.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 2490.30 | 2439.73 | 2425.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 2371.10 | 2439.18 | 2431.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:45:00 | 2379.20 | 2439.18 | 2431.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 2378.60 | 2427.06 | 2426.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 2373.00 | 2427.06 | 2426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 2414.90 | 2424.63 | 2425.45 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 2442.70 | 2429.06 | 2427.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 2489.00 | 2442.35 | 2433.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 2517.00 | 2521.99 | 2493.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 13:00:00 | 2517.00 | 2521.99 | 2493.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2680.30 | 2662.01 | 2634.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2712.00 | 2662.01 | 2634.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 15:15:00 | 2829.00 | 2840.10 | 2841.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 15:15:00 | 2829.00 | 2840.10 | 2841.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2785.60 | 2829.20 | 2836.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2834.40 | 2805.99 | 2817.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 2839.10 | 2805.99 | 2817.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2841.00 | 2812.99 | 2819.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 2841.00 | 2812.99 | 2819.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 2814.50 | 2814.78 | 2819.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 2802.00 | 2812.95 | 2817.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 14:15:00 | 2800.00 | 2812.95 | 2817.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 13:15:00 | 2827.50 | 2800.16 | 2799.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 2827.50 | 2800.16 | 2799.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 2834.50 | 2811.66 | 2805.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 2932.00 | 2943.03 | 2905.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 10:00:00 | 2932.00 | 2943.03 | 2905.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 2933.00 | 2959.21 | 2935.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 2882.90 | 2959.21 | 2935.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 2942.50 | 2955.87 | 2936.42 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 2889.80 | 2925.98 | 2927.65 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 2961.00 | 2925.39 | 2922.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2981.00 | 2946.50 | 2933.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-26 13:15:00 | 1294.60 | 2025-05-27 09:15:00 | 1282.30 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1303.00 | 2025-06-06 09:15:00 | 1433.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1302.50 | 2025-06-06 09:15:00 | 1432.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-14 11:45:00 | 1641.00 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-14 14:00:00 | 1638.60 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-07-14 15:00:00 | 1641.00 | 2025-07-15 12:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-22 09:30:00 | 1643.50 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-07-22 10:45:00 | 1640.60 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-23 14:15:00 | 1643.90 | 2025-07-23 15:15:00 | 1652.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1558.00 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2025-07-30 12:00:00 | 1560.60 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-07-30 13:30:00 | 1560.30 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-07-31 13:30:00 | 1558.70 | 2025-08-04 09:15:00 | 1595.70 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-08-07 13:30:00 | 1543.80 | 2025-08-07 14:15:00 | 1576.00 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-08-14 12:15:00 | 1630.80 | 2025-08-20 13:15:00 | 1643.40 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-08-14 14:45:00 | 1632.00 | 2025-08-20 13:15:00 | 1643.40 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1603.00 | 2025-08-28 09:15:00 | 1522.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:15:00 | 1603.00 | 2025-09-01 09:15:00 | 1538.00 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest2 | 2025-09-04 12:45:00 | 1544.70 | 2025-09-04 14:15:00 | 1530.20 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-09-11 14:30:00 | 1514.60 | 2025-09-12 09:15:00 | 1545.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-22 10:45:00 | 1615.80 | 2025-09-23 09:15:00 | 1583.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-22 13:00:00 | 1610.20 | 2025-09-23 09:15:00 | 1583.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-23 14:15:00 | 1611.60 | 2025-09-24 15:15:00 | 1589.30 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-24 09:15:00 | 1611.10 | 2025-09-24 15:15:00 | 1589.30 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-09-26 09:15:00 | 1608.60 | 2025-09-26 11:15:00 | 1597.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-26 09:45:00 | 1617.00 | 2025-09-26 11:15:00 | 1597.80 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-09 09:45:00 | 1652.40 | 2025-10-14 09:15:00 | 1817.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-09 10:30:00 | 1659.90 | 2025-10-14 09:15:00 | 1825.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-24 09:15:00 | 1841.30 | 2025-10-27 13:15:00 | 1859.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1912.00 | 2025-11-12 11:15:00 | 1877.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 09:45:00 | 1916.00 | 2025-11-12 11:15:00 | 1877.20 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-11-19 10:00:00 | 1951.60 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-11-19 11:15:00 | 1950.50 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-11-21 12:00:00 | 1954.00 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-21 13:45:00 | 1950.00 | 2025-11-21 14:15:00 | 1936.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-03 09:15:00 | 2055.60 | 2025-12-03 09:15:00 | 2039.80 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-03 11:00:00 | 2046.40 | 2025-12-03 14:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-03 11:30:00 | 2048.80 | 2025-12-03 14:15:00 | 2029.40 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-04 09:30:00 | 2056.80 | 2025-12-04 10:15:00 | 2018.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2041.80 | 2025-12-17 11:15:00 | 2020.20 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-12-17 09:45:00 | 2039.80 | 2025-12-17 11:15:00 | 2020.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-30 10:15:00 | 2204.00 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-31 09:15:00 | 2219.40 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-01-01 09:45:00 | 2207.40 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-01 15:15:00 | 2203.00 | 2026-01-05 13:15:00 | 2188.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-01-12 11:15:00 | 2197.00 | 2026-01-13 09:15:00 | 2291.00 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2026-01-23 12:00:00 | 2312.00 | 2026-01-27 09:15:00 | 2390.00 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2026-01-23 12:30:00 | 2313.00 | 2026-01-27 09:15:00 | 2390.00 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2411.40 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-03 11:15:00 | 2408.30 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-02-03 12:00:00 | 2402.40 | 2026-02-03 13:15:00 | 2427.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-17 09:15:00 | 2311.20 | 2026-02-18 15:15:00 | 2337.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-20 09:15:00 | 2398.00 | 2026-03-11 14:15:00 | 2525.00 | STOP_HIT | 1.00 | 5.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 2479.70 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-03-13 11:00:00 | 2480.90 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2026-03-13 12:30:00 | 2483.10 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-13 13:30:00 | 2485.80 | 2026-03-16 14:15:00 | 2549.50 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2712.00 | 2026-04-21 15:15:00 | 2829.00 | STOP_HIT | 1.00 | 4.31% |
| SELL | retest2 | 2026-04-23 13:30:00 | 2802.00 | 2026-04-27 13:15:00 | 2827.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-23 14:15:00 | 2800.00 | 2026-04-27 13:15:00 | 2827.50 | STOP_HIT | 1.00 | -0.98% |
