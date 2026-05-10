# Inventurus Knowledge Solutions Ltd. (IKS)

## Backtest Summary

- **Window:** 2024-12-19 09:15:00 → 2026-05-08 15:15:00 (2389 bars)
- **Last close:** 1686.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 48 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 42 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 47 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 33
- **Target hits / Stop hits / Partials:** 4 / 38 / 5
- **Avg / median % per leg:** -0.86% / -2.23%
- **Sum % (uncompounded):** -40.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 4 | 20 | 0 | -0.98% | -23.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 4 | 20 | 0 | -0.98% | -23.6% |
| SELL (all) | 23 | 10 | 43.5% | 0 | 18 | 5 | -0.74% | -16.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 10 | 43.5% | 0 | 18 | 5 | -0.74% | -16.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 47 | 14 | 29.8% | 4 | 38 | 5 | -0.86% | -40.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 14:15:00 | 1728.20 | 1585.55 | 1585.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 10:15:00 | 1742.00 | 1589.70 | 1587.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 14:15:00 | 1625.00 | 1628.46 | 1608.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 15:00:00 | 1625.00 | 1628.46 | 1608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1609.50 | 1628.28 | 1608.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 1604.00 | 1628.28 | 1608.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1620.00 | 1628.19 | 1608.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 1638.10 | 1619.92 | 1609.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 1637.90 | 1620.38 | 1609.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 1638.90 | 1620.78 | 1609.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 1593.00 | 1620.51 | 1609.97 | SL hit (close<static) qty=1.00 sl=1597.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 1593.00 | 1620.51 | 1609.97 | SL hit (close<static) qty=1.00 sl=1597.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-04 14:15:00 | 1593.00 | 1620.51 | 1609.97 | SL hit (close<static) qty=1.00 sl=1597.80 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:45:00 | 1640.00 | 1614.40 | 1607.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1611.70 | 1614.61 | 1607.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 1611.70 | 1614.61 | 1607.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.80 | 1614.41 | 1607.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1599.80 | 1614.41 | 1607.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1594.00 | 1614.21 | 1607.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-10 10:15:00 | 1594.00 | 1614.21 | 1607.76 | SL hit (close<static) qty=1.00 sl=1597.80 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1594.20 | 1614.21 | 1607.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1594.80 | 1611.76 | 1606.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 1594.80 | 1611.76 | 1606.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1596.20 | 1610.16 | 1606.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 1588.50 | 1610.16 | 1606.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1601.80 | 1610.08 | 1606.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 1591.30 | 1610.08 | 1606.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1601.30 | 1609.99 | 1606.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1601.30 | 1609.99 | 1606.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1612.50 | 1610.02 | 1606.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1601.00 | 1610.02 | 1606.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1611.70 | 1610.04 | 1606.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 1610.30 | 1610.04 | 1606.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1619.40 | 1610.13 | 1606.42 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 13:15:00 | 1581.50 | 1603.98 | 1603.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 12:15:00 | 1571.80 | 1602.61 | 1603.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 13:15:00 | 1601.10 | 1601.02 | 1602.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 1601.10 | 1601.02 | 1602.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1601.10 | 1601.02 | 1602.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 1597.00 | 1600.83 | 1602.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:45:00 | 1590.20 | 1599.90 | 1601.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 15:15:00 | 1592.50 | 1599.91 | 1601.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 09:30:00 | 1596.50 | 1597.99 | 1600.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 1605.50 | 1595.01 | 1599.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 1610.00 | 1595.01 | 1599.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 1590.00 | 1594.96 | 1599.02 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1622.90 | 1595.24 | 1599.14 | SL hit (close>static) qty=1.00 sl=1612.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1622.90 | 1595.24 | 1599.14 | SL hit (close>static) qty=1.00 sl=1612.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1622.90 | 1595.24 | 1599.14 | SL hit (close>static) qty=1.00 sl=1612.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1622.90 | 1595.24 | 1599.14 | SL hit (close>static) qty=1.00 sl=1612.20 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1616.50 | 1595.24 | 1599.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1605.10 | 1595.34 | 1599.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 1599.50 | 1597.23 | 1599.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 1599.00 | 1597.35 | 1599.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1598.90 | 1597.31 | 1599.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 10:00:00 | 1596.20 | 1589.99 | 1595.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1603.30 | 1590.13 | 1595.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:00:00 | 1603.30 | 1590.13 | 1595.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1599.00 | 1590.22 | 1595.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 11:45:00 | 1600.90 | 1590.22 | 1595.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1605.40 | 1590.37 | 1595.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:00:00 | 1605.40 | 1590.37 | 1595.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 1583.20 | 1590.37 | 1595.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:00:00 | 1574.00 | 1590.20 | 1595.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 10:45:00 | 1573.50 | 1589.85 | 1595.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 1613.50 | 1589.52 | 1594.87 | SL hit (close>static) qty=1.00 sl=1598.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 1613.50 | 1589.52 | 1594.87 | SL hit (close>static) qty=1.00 sl=1598.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1573.20 | 1589.53 | 1594.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 1574.50 | 1589.36 | 1594.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1600.10 | 1586.48 | 1592.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 1600.10 | 1586.48 | 1592.87 | SL hit (close>static) qty=1.00 sl=1598.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-25 14:15:00 | 1600.10 | 1586.48 | 1592.87 | SL hit (close>static) qty=1.00 sl=1598.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-08-25 15:00:00 | 1600.10 | 1586.48 | 1592.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1600.00 | 1586.62 | 1592.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1615.00 | 1586.62 | 1592.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1614.20 | 1586.89 | 1593.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 14:45:00 | 1596.10 | 1587.52 | 1593.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1519.52 | 1584.41 | 1591.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1519.05 | 1584.41 | 1591.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1518.95 | 1584.41 | 1591.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1516.39 | 1584.41 | 1591.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1516.29 | 1584.41 | 1591.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1570.90 | 1569.26 | 1582.41 | SL hit (close>ema200) qty=0.50 sl=1569.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1570.90 | 1569.26 | 1582.41 | SL hit (close>ema200) qty=0.50 sl=1569.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1570.90 | 1569.26 | 1582.41 | SL hit (close>ema200) qty=0.50 sl=1569.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1570.90 | 1569.26 | 1582.41 | SL hit (close>ema200) qty=0.50 sl=1569.26 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 14:15:00 | 1570.90 | 1569.26 | 1582.41 | SL hit (close>ema200) qty=0.50 sl=1569.26 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1651.70 | 1552.18 | 1551.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 15:15:00 | 1672.40 | 1555.37 | 1553.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 09:15:00 | 1590.80 | 1593.72 | 1576.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:30:00 | 1583.80 | 1593.72 | 1576.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1590.00 | 1593.81 | 1578.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 1585.00 | 1593.81 | 1578.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1577.70 | 1593.65 | 1578.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 1577.30 | 1593.65 | 1578.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1584.50 | 1593.56 | 1578.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 1580.20 | 1593.56 | 1578.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1573.00 | 1593.18 | 1578.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1573.00 | 1593.18 | 1578.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 1561.20 | 1592.86 | 1578.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 1586.90 | 1592.86 | 1578.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 10:00:00 | 1580.80 | 1631.28 | 1606.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:00:00 | 1578.80 | 1628.75 | 1605.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1576.00 | 1625.55 | 1604.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-22 12:15:00 | 1745.59 | 1639.46 | 1617.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-22 12:15:00 | 1738.88 | 1639.46 | 1617.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-22 12:15:00 | 1736.68 | 1639.46 | 1617.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2025-12-22 12:15:00 | 1733.60 | 1639.46 | 1617.00 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1660.90 | 1663.68 | 1635.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1670.10 | 1663.75 | 1635.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 1670.20 | 1663.84 | 1635.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 1668.30 | 1671.19 | 1644.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:00:00 | 1672.80 | 1671.21 | 1644.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1646.30 | 1670.55 | 1644.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 1633.70 | 1670.55 | 1644.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1640.00 | 1670.24 | 1644.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 1640.00 | 1670.24 | 1644.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 1650.00 | 1670.04 | 1644.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 12:15:00 | 1650.20 | 1670.04 | 1644.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 10:00:00 | 1650.90 | 1669.35 | 1645.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 10:15:00 | 1626.40 | 1668.92 | 1644.92 | SL hit (close<static) qty=1.00 sl=1638.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 10:15:00 | 1626.40 | 1668.92 | 1644.92 | SL hit (close<static) qty=1.00 sl=1638.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:30:00 | 1654.90 | 1668.54 | 1644.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1613.40 | 1674.16 | 1652.38 | SL hit (close<static) qty=1.00 sl=1623.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1613.40 | 1674.16 | 1652.38 | SL hit (close<static) qty=1.00 sl=1623.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1613.40 | 1674.16 | 1652.38 | SL hit (close<static) qty=1.00 sl=1623.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1613.40 | 1674.16 | 1652.38 | SL hit (close<static) qty=1.00 sl=1623.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 10:15:00 | 1613.40 | 1674.16 | 1652.38 | SL hit (close<static) qty=1.00 sl=1638.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-21 12:00:00 | 1652.90 | 1673.95 | 1652.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1651.70 | 1673.73 | 1652.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1651.70 | 1673.73 | 1652.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1650.70 | 1673.50 | 1652.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 1650.80 | 1673.50 | 1652.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1656.90 | 1673.33 | 1652.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 1687.80 | 1673.47 | 1652.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1683.80 | 1673.55 | 1652.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 1684.90 | 1673.61 | 1652.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 14:15:00 | 1683.00 | 1673.79 | 1653.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1658.30 | 1673.81 | 1653.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 1658.30 | 1673.81 | 1653.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1650.00 | 1673.57 | 1653.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 11:00:00 | 1650.00 | 1673.57 | 1653.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 1640.50 | 1673.25 | 1653.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1640.50 | 1673.25 | 1653.47 | SL hit (close<static) qty=1.00 sl=1646.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1640.50 | 1673.25 | 1653.47 | SL hit (close<static) qty=1.00 sl=1646.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1640.50 | 1673.25 | 1653.47 | SL hit (close<static) qty=1.00 sl=1646.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1640.50 | 1673.25 | 1653.47 | SL hit (close<static) qty=1.00 sl=1646.60 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 1640.50 | 1673.25 | 1653.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1645.00 | 1672.14 | 1653.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1624.90 | 1672.14 | 1653.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1606.30 | 1671.48 | 1653.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-27 09:15:00 | 1606.30 | 1671.48 | 1653.07 | SL hit (close<static) qty=1.00 sl=1638.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 1622.80 | 1671.48 | 1653.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 1551.80 | 1670.29 | 1652.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 1551.80 | 1670.29 | 1652.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1604.50 | 1665.30 | 1650.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1606.40 | 1665.30 | 1650.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1560.30 | 1664.26 | 1650.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:45:00 | 1569.60 | 1664.26 | 1650.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1573.50 | 1637.73 | 1637.78 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1704.30 | 1637.98 | 1637.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1712.00 | 1642.34 | 1640.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1630.30 | 1657.72 | 1648.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1630.30 | 1657.72 | 1648.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1630.30 | 1657.72 | 1648.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 15:15:00 | 1665.00 | 1656.84 | 1648.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:00:00 | 1663.30 | 1652.41 | 1646.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 1664.10 | 1652.51 | 1647.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:45:00 | 1663.30 | 1652.61 | 1647.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 1639.70 | 1652.57 | 1647.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 1639.80 | 1652.57 | 1647.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1630.50 | 1652.35 | 1647.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 1630.50 | 1652.35 | 1647.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1580.40 | 1650.70 | 1646.40 | SL hit (close<static) qty=1.00 sl=1588.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1580.40 | 1650.70 | 1646.40 | SL hit (close<static) qty=1.00 sl=1588.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1580.40 | 1650.70 | 1646.40 | SL hit (close<static) qty=1.00 sl=1588.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-20 13:15:00 | 1580.40 | 1650.70 | 1646.40 | SL hit (close<static) qty=1.00 sl=1588.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1516.30 | 1641.31 | 1641.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 1480.50 | 1639.71 | 1641.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 09:15:00 | 1392.20 | 1391.33 | 1465.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-07 09:30:00 | 1383.90 | 1391.33 | 1465.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1456.10 | 1393.19 | 1463.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 15:00:00 | 1428.60 | 1436.28 | 1471.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 09:45:00 | 1429.50 | 1436.14 | 1470.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1416.80 | 1436.38 | 1470.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 11:00:00 | 1425.00 | 1436.15 | 1469.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-24 09:15:00 | 1495.00 | 1435.91 | 1467.37 | SL hit (close>static) qty=1.00 sl=1468.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 1458.00 | 1435.95 | 1467.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1593.60 | 1452.94 | 1472.15 | SL hit (close>static) qty=1.00 sl=1573.80 alert=retest2 |

### Cycle 7 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 1696.10 | 1489.92 | 1489.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 1704.70 | 1509.77 | 1499.92 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-03 09:45:00 | 1638.10 | 2025-07-04 14:15:00 | 1593.00 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-07-03 11:30:00 | 1637.90 | 2025-07-04 14:15:00 | 1593.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-07-03 13:30:00 | 1638.90 | 2025-07-04 14:15:00 | 1593.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2025-07-09 10:45:00 | 1640.00 | 2025-07-10 10:15:00 | 1594.00 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-07-31 14:30:00 | 1597.00 | 2025-08-07 09:15:00 | 1622.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-01 11:45:00 | 1590.20 | 2025-08-07 09:15:00 | 1622.90 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-08-01 15:15:00 | 1592.50 | 2025-08-07 09:15:00 | 1622.90 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-08-05 09:30:00 | 1596.50 | 2025-08-07 09:15:00 | 1622.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-08 15:00:00 | 1599.50 | 2025-08-21 11:15:00 | 1613.50 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-08-11 09:30:00 | 1599.00 | 2025-08-21 11:15:00 | 1613.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-11 10:30:00 | 1598.90 | 2025-08-25 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-08-18 10:00:00 | 1596.20 | 2025-08-25 14:15:00 | 1600.10 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-08-19 14:00:00 | 1574.00 | 2025-08-29 09:15:00 | 1519.52 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-08-20 10:45:00 | 1573.50 | 2025-08-29 09:15:00 | 1519.05 | PARTIAL | 0.50 | 3.46% |
| SELL | retest2 | 2025-08-22 09:30:00 | 1573.20 | 2025-08-29 09:15:00 | 1518.95 | PARTIAL | 0.50 | 3.45% |
| SELL | retest2 | 2025-08-22 10:30:00 | 1574.50 | 2025-08-29 09:15:00 | 1516.39 | PARTIAL | 0.50 | 3.69% |
| SELL | retest2 | 2025-08-26 14:45:00 | 1596.10 | 2025-08-29 09:15:00 | 1516.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-19 14:00:00 | 1574.00 | 2025-09-03 14:15:00 | 1570.90 | STOP_HIT | 0.50 | 0.20% |
| SELL | retest2 | 2025-08-20 10:45:00 | 1573.50 | 2025-09-03 14:15:00 | 1570.90 | STOP_HIT | 0.50 | 0.17% |
| SELL | retest2 | 2025-08-22 09:30:00 | 1573.20 | 2025-09-03 14:15:00 | 1570.90 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2025-08-22 10:30:00 | 1574.50 | 2025-09-03 14:15:00 | 1570.90 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-08-26 14:45:00 | 1596.10 | 2025-09-03 14:15:00 | 1570.90 | STOP_HIT | 0.50 | 1.58% |
| BUY | retest2 | 2025-11-24 09:15:00 | 1586.90 | 2025-12-22 12:15:00 | 1745.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 10:00:00 | 1580.80 | 2025-12-22 12:15:00 | 1738.88 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-09 15:00:00 | 1578.80 | 2025-12-22 12:15:00 | 1736.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1576.00 | 2025-12-22 12:15:00 | 1733.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 10:00:00 | 1670.10 | 2026-01-12 10:15:00 | 1626.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-01-01 10:45:00 | 1670.20 | 2026-01-12 10:15:00 | 1626.40 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2026-01-08 12:15:00 | 1668.30 | 2026-01-21 10:15:00 | 1613.40 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-01-08 13:00:00 | 1672.80 | 2026-01-21 10:15:00 | 1613.40 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2026-01-09 12:15:00 | 1650.20 | 2026-01-21 10:15:00 | 1613.40 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-01-12 10:00:00 | 1650.90 | 2026-01-21 10:15:00 | 1613.40 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2026-01-12 12:30:00 | 1654.90 | 2026-01-21 10:15:00 | 1613.40 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-01-21 12:00:00 | 1652.90 | 2026-01-23 11:15:00 | 1640.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-01-22 09:30:00 | 1687.80 | 2026-01-23 11:15:00 | 1640.50 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-01-22 11:15:00 | 1683.80 | 2026-01-23 11:15:00 | 1640.50 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2026-01-22 12:15:00 | 1684.90 | 2026-01-23 11:15:00 | 1640.50 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-01-22 14:15:00 | 1683.00 | 2026-01-27 09:15:00 | 1606.30 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest2 | 2026-02-13 15:15:00 | 1665.00 | 2026-02-20 13:15:00 | 1580.40 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2026-02-19 11:00:00 | 1663.30 | 2026-02-20 13:15:00 | 1580.40 | STOP_HIT | 1.00 | -4.98% |
| BUY | retest2 | 2026-02-19 11:30:00 | 1664.10 | 2026-02-20 13:15:00 | 1580.40 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest2 | 2026-02-19 12:45:00 | 1663.30 | 2026-02-20 13:15:00 | 1580.40 | STOP_HIT | 1.00 | -4.98% |
| SELL | retest2 | 2026-04-20 15:00:00 | 1428.60 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.65% |
| SELL | retest2 | 2026-04-21 09:45:00 | 1429.50 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 1416.80 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2026-04-22 11:00:00 | 1425.00 | 2026-04-24 09:15:00 | 1495.00 | STOP_HIT | 1.00 | -4.91% |
| SELL | retest2 | 2026-04-24 10:30:00 | 1458.00 | 2026-04-30 09:15:00 | 1593.60 | STOP_HIT | 1.00 | -9.30% |
