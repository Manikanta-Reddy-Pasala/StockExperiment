# Bajaj Finserv Ltd. (BAJAJFINSV)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1814.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 15 |
| ALERT2 | 15 |
| ALERT2_SKIP | 8 |
| ALERT3 | 107 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 87 |
| PARTIAL | 14 |
| TARGET_HIT | 17 |
| STOP_HIT | 69 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 43 / 57
- **Target hits / Stop hits / Partials:** 17 / 69 / 14
- **Avg / median % per leg:** 1.66% / -0.67%
- **Sum % (uncompounded):** 166.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 15 | 31.2% | 13 | 35 | 0 | 1.60% | 76.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 48 | 15 | 31.2% | 13 | 35 | 0 | 1.60% | 76.7% |
| SELL (all) | 52 | 28 | 53.8% | 4 | 34 | 14 | 1.72% | 89.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 28 | 53.8% | 4 | 34 | 14 | 1.72% | 89.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 100 | 43 | 43.0% | 17 | 69 | 14 | 1.66% | 166.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-15 13:15:00 | 1426.55 | 1359.57 | 1359.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-16 09:15:00 | 1433.00 | 1361.49 | 1360.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 10:15:00 | 1577.10 | 1585.90 | 1530.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-28 10:30:00 | 1574.50 | 1585.90 | 1530.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 1529.00 | 1583.72 | 1535.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 13:45:00 | 1531.80 | 1583.72 | 1535.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 14:15:00 | 1527.10 | 1583.16 | 1535.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-02 15:00:00 | 1527.10 | 1583.16 | 1535.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 10:15:00 | 1512.65 | 1515.03 | 1513.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 11:15:00 | 1517.00 | 1515.03 | 1513.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 14:15:00 | 1499.15 | 1514.88 | 1513.65 | SL hit (close<static) qty=1.00 sl=1506.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 1620.95 | 1637.77 | 1637.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 14:15:00 | 1619.30 | 1637.59 | 1637.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 1642.05 | 1637.51 | 1637.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 1642.05 | 1637.51 | 1637.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 1642.05 | 1637.51 | 1637.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:30:00 | 1643.15 | 1637.51 | 1637.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1655.55 | 1637.69 | 1637.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 11:00:00 | 1655.55 | 1637.69 | 1637.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 1660.25 | 1637.91 | 1637.88 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 14:15:00 | 1613.35 | 1637.82 | 1637.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 11:15:00 | 1596.65 | 1636.71 | 1637.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 1613.20 | 1608.71 | 1621.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-19 11:30:00 | 1610.20 | 1608.71 | 1621.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1619.10 | 1608.92 | 1620.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 14:30:00 | 1620.95 | 1608.92 | 1620.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 1618.60 | 1609.01 | 1620.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 09:15:00 | 1603.35 | 1609.01 | 1620.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 14:00:00 | 1616.95 | 1606.39 | 1618.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 15:00:00 | 1616.30 | 1606.49 | 1618.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 11:00:00 | 1617.10 | 1606.82 | 1618.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 11:15:00 | 1620.65 | 1606.95 | 1618.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 11:45:00 | 1621.70 | 1606.95 | 1618.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 1623.95 | 1607.12 | 1618.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-26 13:00:00 | 1623.95 | 1607.12 | 1618.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 13:15:00 | 1620.20 | 1607.25 | 1618.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 14:15:00 | 1619.50 | 1607.25 | 1618.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 12:45:00 | 1617.00 | 1607.63 | 1618.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 10:00:00 | 1618.95 | 1606.56 | 1615.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1536.10 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1535.48 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1536.24 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1538.52 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1536.15 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-06 09:15:00 | 1538.00 | 1604.50 | 1614.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-11 09:15:00 | 1600.20 | 1599.72 | 1611.16 | SL hit (close>ema200) qty=0.50 sl=1599.72 alert=retest2 |

### Cycle 5 — BUY (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 13:15:00 | 1679.45 | 1611.51 | 1611.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1692.05 | 1613.61 | 1612.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1628.35 | 1636.26 | 1624.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1628.35 | 1636.26 | 1624.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1628.35 | 1636.26 | 1624.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 1628.35 | 1636.26 | 1624.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1620.25 | 1636.04 | 1624.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:00:00 | 1620.25 | 1636.04 | 1624.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 12:15:00 | 1619.10 | 1635.87 | 1624.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 12:30:00 | 1614.00 | 1635.87 | 1624.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 1621.80 | 1635.65 | 1624.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 14:45:00 | 1617.95 | 1635.65 | 1624.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 1618.35 | 1635.47 | 1624.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 09:15:00 | 1624.25 | 1635.47 | 1624.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 10:15:00 | 1624.05 | 1635.18 | 1624.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 10:30:00 | 1616.65 | 1635.18 | 1624.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 1622.80 | 1635.06 | 1624.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:45:00 | 1624.35 | 1635.06 | 1624.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 1619.15 | 1634.90 | 1624.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-18 13:00:00 | 1619.15 | 1634.90 | 1624.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 13:15:00 | 1608.85 | 1634.64 | 1624.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 14:15:00 | 1624.65 | 1631.84 | 1623.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 1631.10 | 1631.59 | 1623.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 14:00:00 | 1624.00 | 1631.39 | 1623.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 10:00:00 | 1628.00 | 1631.77 | 1624.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1629.10 | 1631.74 | 1624.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 1624.35 | 1631.74 | 1624.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1604.60 | 1632.44 | 1625.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-26 11:15:00 | 1597.85 | 1631.79 | 1624.82 | SL hit (close<static) qty=1.00 sl=1599.15 alert=retest2 |

### Cycle 6 — SELL (started 2024-05-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 11:15:00 | 1572.30 | 1619.70 | 1619.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 12:15:00 | 1567.85 | 1619.19 | 1619.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 1608.45 | 1602.39 | 1609.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 12:15:00 | 1608.45 | 1602.39 | 1609.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 1608.45 | 1602.39 | 1609.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:00:00 | 1608.45 | 1602.39 | 1609.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 1610.85 | 1602.47 | 1609.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 1610.85 | 1602.47 | 1609.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 14:15:00 | 1608.80 | 1602.54 | 1609.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:45:00 | 1602.75 | 1602.81 | 1609.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 1600.25 | 1602.83 | 1609.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 1602.95 | 1602.80 | 1609.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 1603.00 | 1602.81 | 1609.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1604.85 | 1602.83 | 1609.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:30:00 | 1609.10 | 1602.83 | 1609.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1522.61 | 1596.48 | 1605.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1520.24 | 1596.48 | 1605.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1522.80 | 1596.48 | 1605.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1522.85 | 1596.48 | 1605.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 1442.48 | 1587.31 | 1599.96 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 1646.80 | 1592.23 | 1592.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1665.20 | 1597.96 | 1595.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1594.50 | 1606.06 | 1600.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:30:00 | 1590.65 | 1606.06 | 1600.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1575.30 | 1605.76 | 1599.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 11:00:00 | 1575.30 | 1605.76 | 1599.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1560.85 | 1594.84 | 1594.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.19 | 1592.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.17 | 1586.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 1581.00 | 1580.17 | 1586.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.27 | 1586.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 1590.75 | 1580.27 | 1586.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 1591.15 | 1580.38 | 1586.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:30:00 | 1591.45 | 1580.38 | 1586.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1613.55 | 1580.71 | 1587.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 1613.55 | 1580.71 | 1587.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1602.65 | 1580.93 | 1587.13 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.68 | 1592.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.55 | 1593.54 | 1592.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.73 | 1782.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 1850.75 | 1860.73 | 1782.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.00 | 1859.52 | 1800.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:30:00 | 1823.50 | 1859.14 | 1800.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 1821.95 | 1859.14 | 1800.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 1775.35 | 1855.34 | 1800.72 | SL hit (close<static) qty=1.00 sl=1793.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1679.60 | 1770.44 | 1770.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1675.50 | 1769.50 | 1770.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.20 | 1699.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 09:45:00 | 1663.35 | 1661.20 | 1699.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.86 | 1659.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:00:00 | 1657.65 | 1620.86 | 1659.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 1711.90 | 1621.76 | 1659.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:45:00 | 1712.40 | 1621.76 | 1659.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 13:15:00 | 1700.45 | 1622.55 | 1660.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:30:00 | 1687.45 | 1631.69 | 1662.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 12:15:00 | 1688.55 | 1631.69 | 1662.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 13:00:00 | 1688.00 | 1632.25 | 1662.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 14:15:00 | 1690.15 | 1636.61 | 1663.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1694.25 | 1650.89 | 1667.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 1708.20 | 1650.89 | 1667.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 1727.00 | 1654.39 | 1668.89 | SL hit (close>static) qty=1.00 sl=1714.55 alert=retest2 |

### Cycle 11 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.28 | 1679.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.54 | 1682.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1693.40 | 1704.55 | 1693.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1751.55 | 1705.01 | 1693.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1772.55 | 1705.01 | 1693.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:45:00 | 1762.55 | 1705.46 | 1693.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 1769.00 | 1705.46 | 1693.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:00:00 | 1760.85 | 1737.71 | 1714.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-27 10:15:00 | 1936.93 | 1804.07 | 1760.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1942.00 | 1992.10 | 1992.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1928.90 | 1991.47 | 1992.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.58 | 1976.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 2009.00 | 1963.58 | 1976.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 2003.90 | 1963.98 | 1976.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 2000.40 | 1965.19 | 1976.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 2000.10 | 1965.44 | 1976.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 15:00:00 | 2000.90 | 1960.54 | 1969.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 2017.60 | 1963.44 | 1971.11 | SL hit (close>static) qty=1.00 sl=2017.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.48 | 2006.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 2015.00 | 2025.48 | 2006.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.09 | 2006.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 12:00:00 | 2001.40 | 2025.09 | 2006.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 12:15:00 | 2004.50 | 2024.89 | 2006.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 2017.40 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 2012.70 | 2024.05 | 2006.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1999.30 | 2023.59 | 2006.07 | SL hit (close<static) qty=1.00 sl=2000.50 alert=retest2 |

### Cycle 14 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2009.50 | 2053.54 | 2053.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.66 | 2052.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.18 | 2049.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:00:00 | 2073.20 | 2046.18 | 2049.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2062.40 | 2046.34 | 2049.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:30:00 | 2074.20 | 2046.34 | 2049.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 2049.70 | 2046.40 | 2049.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2043.80 | 2046.40 | 2049.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:00:00 | 2044.50 | 2046.38 | 2049.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 2023.10 | 2046.38 | 2049.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1941.61 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 1942.27 | 2024.09 | 2035.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 13:15:00 | 1921.94 | 2010.93 | 2027.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2010.50 | 1986.60 | 2011.45 | SL hit (close>ema200) qty=0.50 sl=1986.60 alert=retest2 |

### Cycle 15 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 2016.50 | 2024.11 | 2023.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 2020.60 | 2024.07 | 2023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:15:00 | 2016.10 | 2024.07 | 2023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 2017.00 | 2024.00 | 2023.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:30:00 | 2013.10 | 2024.00 | 2023.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1966.20 | 2022.51 | 2022.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 1946.20 | 2021.16 | 2021.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.43 | 1865.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1805.00 | 1783.43 | 1865.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1852.40 | 1797.68 | 1852.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 12:45:00 | 1852.00 | 1797.68 | 1852.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1798.19 | 1852.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 1849.40 | 1798.19 | 1852.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1850.40 | 1798.71 | 1852.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 1850.10 | 1798.71 | 1852.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1850.10 | 1799.22 | 1852.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1854.00 | 1799.22 | 1852.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1840.80 | 1799.64 | 1852.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1827.80 | 1802.48 | 1852.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 14:15:00 | 1736.41 | 1793.15 | 1838.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1790.06 | 1834.08 | SL hit (close>ema200) qty=0.50 sl=1790.06 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-30 11:15:00 | 1517.00 | 2023-08-30 14:15:00 | 1499.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-09-01 15:15:00 | 1514.50 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-09-05 09:30:00 | 1516.00 | 2023-09-06 12:15:00 | 1506.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2023-09-05 14:00:00 | 1515.00 | 2023-10-17 09:15:00 | 1665.95 | TARGET_HIT | 1.00 | 9.96% |
| BUY | retest2 | 2023-09-06 11:15:00 | 1514.55 | 2023-10-17 09:15:00 | 1667.60 | TARGET_HIT | 1.00 | 10.11% |
| BUY | retest2 | 2023-09-06 12:15:00 | 1514.05 | 2023-10-17 09:15:00 | 1666.50 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2023-09-06 15:00:00 | 1518.75 | 2023-10-17 09:15:00 | 1670.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 10:30:00 | 1515.10 | 2023-10-17 09:15:00 | 1666.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 13:30:00 | 1515.25 | 2023-10-17 09:15:00 | 1666.78 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-07 14:15:00 | 1515.90 | 2023-10-17 09:15:00 | 1667.49 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-20 09:15:00 | 1603.35 | 2024-03-06 09:15:00 | 1536.10 | PARTIAL | 0.50 | 4.19% |
| SELL | retest2 | 2024-02-23 14:00:00 | 1616.95 | 2024-03-06 09:15:00 | 1535.48 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2024-02-23 15:00:00 | 1616.30 | 2024-03-06 09:15:00 | 1536.24 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2024-02-26 11:00:00 | 1617.10 | 2024-03-06 09:15:00 | 1538.52 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2024-02-26 14:15:00 | 1619.50 | 2024-03-06 09:15:00 | 1536.15 | PARTIAL | 0.50 | 5.15% |
| SELL | retest2 | 2024-02-27 12:45:00 | 1617.00 | 2024-03-06 09:15:00 | 1538.00 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-02-20 09:15:00 | 1603.35 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2024-02-23 14:00:00 | 1616.95 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2024-02-23 15:00:00 | 1616.30 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 1.00% |
| SELL | retest2 | 2024-02-26 11:00:00 | 1617.10 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 1.05% |
| SELL | retest2 | 2024-02-26 14:15:00 | 1619.50 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 1.19% |
| SELL | retest2 | 2024-02-27 12:45:00 | 1617.00 | 2024-03-11 09:15:00 | 1600.20 | STOP_HIT | 0.50 | 1.04% |
| SELL | retest2 | 2024-03-05 10:00:00 | 1618.95 | 2024-03-28 09:15:00 | 1636.75 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-04-19 14:15:00 | 1624.65 | 2024-04-26 11:15:00 | 1597.85 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-04-22 09:15:00 | 1631.10 | 2024-04-26 11:15:00 | 1597.85 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-04-22 14:00:00 | 1624.00 | 2024-04-26 11:15:00 | 1597.85 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-04-25 10:00:00 | 1628.00 | 2024-04-26 11:15:00 | 1597.85 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-05-03 09:15:00 | 1697.65 | 2024-05-09 10:15:00 | 1579.45 | STOP_HIT | 1.00 | -6.96% |
| BUY | retest2 | 2024-05-03 10:45:00 | 1650.00 | 2024-05-09 10:15:00 | 1579.45 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2024-05-03 11:30:00 | 1650.15 | 2024-05-09 10:15:00 | 1579.45 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-05-24 14:45:00 | 1602.75 | 2024-05-30 14:15:00 | 1522.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:15:00 | 1600.25 | 2024-05-30 14:15:00 | 1520.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 1602.95 | 2024-05-30 14:15:00 | 1522.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 11:30:00 | 1603.00 | 2024-05-30 14:15:00 | 1522.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 14:45:00 | 1602.75 | 2024-06-04 11:15:00 | 1442.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 09:15:00 | 1600.25 | 2024-06-04 11:15:00 | 1440.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 11:00:00 | 1602.95 | 2024-06-04 11:15:00 | 1442.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-27 11:30:00 | 1603.00 | 2024-06-04 11:15:00 | 1442.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-12 13:45:00 | 1582.50 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-12 15:00:00 | 1578.95 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-06-13 09:30:00 | 1584.00 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-06-13 10:30:00 | 1584.25 | 2024-06-18 09:15:00 | 1597.45 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2024-06-21 15:00:00 | 1572.80 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-06-24 09:15:00 | 1568.45 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-06-25 11:00:00 | 1573.55 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-07-02 09:15:00 | 1573.65 | 2024-07-03 12:15:00 | 1591.45 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-07-03 11:15:00 | 1581.25 | 2024-07-03 14:15:00 | 1594.95 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-05 09:15:00 | 1576.30 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-07-09 15:15:00 | 1581.00 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-10 10:00:00 | 1579.30 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-10-18 10:30:00 | 1823.50 | 2024-10-21 11:15:00 | 1775.35 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1821.95 | 2024-10-21 11:15:00 | 1775.35 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-01-06 11:30:00 | 1687.45 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-01-06 12:15:00 | 1688.55 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-01-06 13:00:00 | 1688.00 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-01-07 14:15:00 | 1690.15 | 2025-01-14 15:15:00 | 1727.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-01-15 13:00:00 | 1670.00 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-01-15 13:30:00 | 1665.45 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-01-15 14:15:00 | 1670.35 | 2025-01-16 09:15:00 | 1684.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1772.55 | 2025-02-27 10:15:00 | 1936.93 | TARGET_HIT | 1.00 | 9.27% |
| BUY | retest2 | 2025-02-01 14:45:00 | 1762.55 | 2025-03-25 09:15:00 | 1938.81 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-01 15:15:00 | 1769.00 | 2025-03-25 09:15:00 | 1945.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-12 11:00:00 | 1760.85 | 2025-03-25 10:15:00 | 1949.81 | TARGET_HIT | 1.00 | 10.73% |
| BUY | retest2 | 2025-03-05 10:30:00 | 1789.10 | 2025-03-26 09:15:00 | 1968.01 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-05 12:15:00 | 1788.35 | 2025-03-26 09:15:00 | 1967.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-18 13:45:00 | 2000.40 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-18 14:30:00 | 2000.10 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-04 15:00:00 | 2000.90 | 2025-09-05 13:15:00 | 2017.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-09-29 09:30:00 | 2017.40 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-29 10:15:00 | 2012.70 | 2025-09-29 11:15:00 | 1999.30 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-29 13:30:00 | 2009.90 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2025.10 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-06 09:15:00 | 2021.40 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-08 12:45:00 | 2019.10 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-10-09 10:30:00 | 2016.20 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-10 12:30:00 | 2013.00 | 2025-10-10 14:15:00 | 2003.40 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-10-13 11:30:00 | 2013.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-10-14 13:00:00 | 2015.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 10:30:00 | 2013.10 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest2 | 2025-11-12 11:45:00 | 2015.90 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | 1.34% |
| BUY | retest2 | 2025-11-14 12:15:00 | 2055.00 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-14 13:45:00 | 2055.20 | 2025-11-19 09:15:00 | 2042.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-18 12:30:00 | 2056.30 | 2025-11-24 11:15:00 | 2038.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-11-18 14:00:00 | 2058.80 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-11-20 10:45:00 | 2065.70 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-11-26 12:00:00 | 2068.50 | 2025-12-03 10:15:00 | 2038.90 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-12-02 10:15:00 | 2066.70 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-02 14:15:00 | 2065.00 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-12-11 14:15:00 | 2069.60 | 2025-12-16 09:15:00 | 2046.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-12 09:15:00 | 2076.70 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -3.24% |
| BUY | retest2 | 2025-12-15 11:15:00 | 2068.90 | 2025-12-29 12:15:00 | 2009.50 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-01-21 10:15:00 | 1941.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-01-21 10:15:00 | 1942.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-01-27 13:15:00 | 1921.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 14:15:00 | 2043.80 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2026-01-06 15:00:00 | 2044.50 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 1.66% |
| SELL | retest2 | 2026-01-07 09:15:00 | 2023.10 | 2026-02-03 09:15:00 | 2010.50 | STOP_HIT | 0.50 | 0.62% |
| SELL | retest2 | 2026-02-11 11:45:00 | 2039.00 | 2026-02-16 15:15:00 | 2053.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-04-30 14:15:00 | 1736.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1827.80 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 0.50 | 2.00% |
