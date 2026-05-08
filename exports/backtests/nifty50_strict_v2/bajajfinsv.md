# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1818.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty booked @ 5% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 10 |
| ALERT3 | 19 |
| PENDING | 63 |
| PENDING_CANCEL | 14 |
| ENTRY1 | 7 |
| ENTRY2 | 42 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 48 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 44
- **Target hits / Stop hits / Partials:** 1 / 47 / 0
- **Avg / median % per leg:** -1.79% / -1.88%
- **Sum % (uncompounded):** -85.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 1 | 20 | 0 | -1.43% | -30.0% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.30% | -17.2% |
| BUY @ 3rd Alert (retest2) | 17 | 4 | 23.5% | 1 | 16 | 0 | -0.76% | -12.8% |
| SELL (all) | 27 | 0 | 0.0% | 0 | 27 | 0 | -2.07% | -55.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.01% | -15.0% |
| SELL @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.70% | -40.9% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -4.60% | -32.2% |
| retest2 (combined) | 41 | 4 | 9.8% | 1 | 40 | 0 | -1.31% | -53.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 10:15:00 | 1610.00 | 1544.83 | 1544.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 14:15:00 | 1620.35 | 1547.65 | 1546.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-26 09:15:00 | 1567.25 | 1592.52 | 1573.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-26 09:15:00 | 1567.25 | 1592.52 | 1573.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 1567.25 | 1592.52 | 1573.80 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-15 10:15:00 | 1606.00 | 1581.05 | 1573.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-15 11:15:00 | 1609.20 | 1581.33 | 1573.48 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-16 09:15:00 | 1601.00 | 1582.08 | 1574.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-16 10:15:00 | 1612.60 | 1582.38 | 1574.25 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-21 10:15:00 | 1603.15 | 1586.83 | 1577.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-21 11:15:00 | 1602.30 | 1586.99 | 1577.55 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-01-23 10:15:00 | 1603.25 | 1654.57 | 1646.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-01-23 11:15:00 | 1589.05 | 1653.92 | 1645.98 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-01-24 09:15:00 | 1602.00 | 1650.77 | 1644.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:15:00 | 1603.70 | 1650.30 | 1644.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 1655.55 | 1640.29 | 1639.95 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-02-02 11:15:00 | 1660.00 | 1640.48 | 1640.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-02 12:15:00 | 1651.05 | 1640.59 | 1640.11 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1610.45 | 1639.67 | 1639.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1610.45 | 1639.67 | 1639.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1610.45 | 1639.67 | 1639.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 09:15:00 | 1610.45 | 1639.67 | 1639.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 09:15:00 | 1610.45 | 1639.67 | 1639.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 11:15:00 | 1597.30 | 1638.99 | 1639.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 1613.20 | 1609.82 | 1622.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 14:15:00 | 1619.10 | 1610.00 | 1622.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1619.10 | 1610.00 | 1622.35 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-02-20 10:15:00 | 1610.00 | 1610.15 | 1622.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-20 11:15:00 | 1616.00 | 1610.21 | 1622.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-20 12:15:00 | 1609.05 | 1610.20 | 1622.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 13:15:00 | 1608.85 | 1610.18 | 1622.08 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-26 15:15:00 | 1614.30 | 1608.14 | 1619.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 1606.05 | 1608.12 | 1619.20 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-02-27 12:15:00 | 1613.10 | 1608.35 | 1619.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:15:00 | 1601.00 | 1608.28 | 1619.06 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-01 10:15:00 | 1612.10 | 1606.18 | 1617.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-01 11:15:00 | 1615.30 | 1606.27 | 1617.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-01 12:15:00 | 1611.30 | 1606.32 | 1616.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 13:15:00 | 1614.00 | 1606.39 | 1616.98 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1605.75 | 1606.51 | 1616.88 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-05 12:15:00 | 1598.40 | 1606.99 | 1616.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:15:00 | 1573.30 | 1606.65 | 1616.40 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 14:15:00 | 1599.60 | 1600.40 | 1612.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 15:15:00 | 1595.00 | 1600.34 | 1611.93 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-21 11:15:00 | 1595.45 | 1591.11 | 1604.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:15:00 | 1600.45 | 1591.20 | 1604.04 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-21 15:15:00 | 1602.60 | 1591.56 | 1604.03 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 1596.20 | 1591.61 | 1603.99 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1611.05 | 1591.74 | 1603.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.76 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.76 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.76 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.76 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-27 12:15:00 | 1585.00 | 1592.69 | 1603.55 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 13:15:00 | 1587.00 | 1592.64 | 1603.46 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.42 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.42 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.42 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.42 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.42 | SL hit (close>static) qty=1.00 sl=1615.60 alert=retest2 |

### Cycle 3 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1675.65 | 1612.35 | 1612.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1692.05 | 1613.79 | 1612.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-25 13:15:00 | 1668.90 | 1632.30 | 1625.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:15:00 | 1654.70 | 1632.53 | 1625.39 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1604.30 | 1632.47 | 1625.44 | SL hit (close<static) qty=1.00 sl=1623.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-03 09:15:00 | 1660.05 | 1626.88 | 1623.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1655.95 | 1627.17 | 1623.46 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-03 15:15:00 | 1623.00 | 1627.15 | 1623.55 | SL hit (close<static) qty=1.00 sl=1623.65 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1570.90 | 1620.16 | 1620.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1562.20 | 1617.23 | 1618.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.34 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-29 10:15:00 | 1579.10 | 1602.35 | 1609.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 1575.15 | 1602.08 | 1609.00 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-12 13:15:00 | 1580.65 | 1574.53 | 1590.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 14:15:00 | 1578.95 | 1574.57 | 1590.18 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-13 10:15:00 | 1583.60 | 1574.82 | 1590.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-13 11:15:00 | 1586.00 | 1574.93 | 1590.05 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 10:15:00 | 1585.05 | 1575.72 | 1590.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 11:15:00 | 1591.85 | 1575.88 | 1590.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-20 09:15:00 | 1579.65 | 1578.51 | 1590.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1580.50 | 1578.53 | 1590.07 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 09:15:00 | 1585.00 | 1578.84 | 1589.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1585.00 | 1578.90 | 1589.86 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1593.60 | 1579.20 | 1589.84 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-06-21 14:15:00 | 1572.80 | 1579.13 | 1589.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1579.90 | 1579.14 | 1589.71 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-25 09:15:00 | 1579.65 | 1579.24 | 1589.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1573.55 | 1579.18 | 1589.26 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1604.75 | 1579.51 | 1589.28 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1604.75 | 1579.51 | 1589.28 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.43 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.43 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.43 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.43 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-01 09:15:00 | 1584.05 | 1584.03 | 1590.58 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 1580.55 | 1583.99 | 1590.53 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-01 13:15:00 | 1579.95 | 1583.93 | 1590.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 1580.00 | 1583.89 | 1590.35 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1586.20 | 1583.18 | 1589.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 1599.00 | 1583.77 | 1589.78 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 1599.00 | 1583.77 | 1589.78 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-05 13:15:00 | 1572.80 | 1583.81 | 1589.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-05 14:15:00 | 1580.20 | 1583.77 | 1589.43 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-08 09:15:00 | 1569.10 | 1583.57 | 1589.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 1568.30 | 1583.42 | 1589.17 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-10 13:15:00 | 1575.00 | 1582.38 | 1588.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-10 14:15:00 | 1581.95 | 1582.38 | 1588.12 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 1593.40 | 1582.79 | 1588.08 | SL hit (close>static) qty=1.00 sl=1592.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-07-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 14:15:00 | 1640.45 | 1592.76 | 1592.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 09:15:00 | 1649.60 | 1593.80 | 1593.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1591.15 | 1597.75 | 1595.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 1591.15 | 1597.75 | 1595.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1591.15 | 1597.75 | 1595.17 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-07-29 14:15:00 | 1618.60 | 1595.20 | 1594.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 1617.15 | 1595.41 | 1594.20 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1575.30 | 1605.87 | 1600.07 | SL hit (close<static) qty=1.00 sl=1591.15 alert=retest2 |

### Cycle 6 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1561.10 | 1594.97 | 1595.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.34 | 1592.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.28 | 1587.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.04 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.74 | 1592.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.30 | 1593.60 | 1593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.78 | 1782.55 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-09 10:15:00 | 1877.30 | 1860.36 | 1784.27 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-09 11:15:00 | 1862.00 | 1860.37 | 1784.66 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-09 12:15:00 | 1871.70 | 1860.48 | 1785.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 13:15:00 | 1877.35 | 1860.65 | 1785.55 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 09:15:00 | 1879.55 | 1860.95 | 1786.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1881.05 | 1861.15 | 1787.29 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 12:15:00 | 1871.15 | 1861.31 | 1788.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 13:15:00 | 1879.55 | 1861.49 | 1788.56 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-11 10:15:00 | 1875.35 | 1861.96 | 1790.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1876.15 | 1862.10 | 1790.67 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.25 | 1859.53 | 1800.28 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-18 10:15:00 | 1821.10 | 1859.15 | 1800.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 1825.65 | 1858.81 | 1800.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | SL hit (close<ema400) qty=1.00 sl=1800.93 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | SL hit (close<ema400) qty=1.00 sl=1800.93 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | SL hit (close<ema400) qty=1.00 sl=1800.93 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | SL hit (close<ema400) qty=1.00 sl=1800.93 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.79 | 1800.93 | SL hit (close<static) qty=1.00 sl=1799.50 alert=retest2 |

### Cycle 8 — SELL (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 10:15:00 | 1679.60 | 1770.94 | 1771.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 11:15:00 | 1675.60 | 1769.99 | 1770.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 09:15:00 | 1661.90 | 1661.43 | 1700.12 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-12-18 09:15:00 | 1626.05 | 1662.88 | 1693.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 10:15:00 | 1630.00 | 1662.56 | 1693.36 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 1657.65 | 1620.89 | 1659.77 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-02 12:15:00 | 1711.80 | 1621.80 | 1660.02 | SL hit (close>ema400) qty=1.00 sl=1660.02 alert=retest1 |

### Cycle 9 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.22 | 1679.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.55 | 1682.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 10:15:00 | 1806.30 | 1815.29 | 1769.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1778.20 | 1813.78 | 1770.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1778.20 | 1813.78 | 1770.52 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 1790.45 | 1813.55 | 1770.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 1786.10 | 1813.28 | 1770.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Target hit | 2025-03-26 09:15:00 | 1964.71 | 1839.47 | 1801.73 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 1940.00 | 1992.63 | 1992.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 10:15:00 | 1916.30 | 1990.75 | 1991.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 2010.90 | 1963.66 | 1976.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 2010.90 | 1963.66 | 1976.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 2010.90 | 1963.66 | 1976.23 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2025-08-26 09:15:00 | 1939.90 | 1966.19 | 1975.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:15:00 | 1943.50 | 1965.96 | 1975.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-02 13:15:00 | 1949.20 | 1957.19 | 1969.09 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-09-02 14:15:00 | 1957.70 | 1957.19 | 1969.04 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-09-04 12:15:00 | 2016.10 | 1959.67 | 1969.60 | SL hit (close>static) qty=1.00 sl=2015.90 alert=retest2 |

### Cycle 11 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.45 | 2006.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 2001.40 | 2025.07 | 2006.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.07 | 2006.19 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-09-29 14:15:00 | 2025.10 | 2023.27 | 2006.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 2024.70 | 2023.29 | 2006.28 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1983.50 | 2022.02 | 2006.30 | SL hit (close<static) qty=1.00 sl=2000.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 2024.00 | 2018.85 | 2005.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 2021.70 | 2018.88 | 2005.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 2016.50 | 2020.85 | 2007.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 2014.30 | 2020.78 | 2007.87 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 2020.20 | 2020.47 | 2007.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 2022.00 | 2020.48 | 2008.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2010.00 | 2020.20 | 2008.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-10-13 14:15:00 | 2019.20 | 2019.34 | 2008.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 2023.50 | 2019.38 | 2008.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 14:15:00 | 2019.00 | 2019.32 | 2008.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 2019.40 | 2019.32 | 2008.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.53 | 2053.09 | SL hit (close<static) qty=1.00 sl=2000.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.53 | 2053.09 | SL hit (close<static) qty=1.00 sl=2000.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.53 | 2053.09 | SL hit (close<static) qty=1.00 sl=2000.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.53 | 2053.09 | SL hit (close<static) qty=1.00 sl=2005.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.53 | 2053.09 | SL hit (close<static) qty=1.00 sl=2005.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-12 12:15:00 | 2028.50 | 2071.25 | 2050.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:15:00 | 2034.50 | 2070.89 | 2050.04 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 13:15:00 | 2017.30 | 2065.76 | 2058.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 2019.90 | 2065.30 | 2058.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 2004.20 | 2054.46 | 2054.07 | SL hit (close<static) qty=1.00 sl=2005.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 2004.20 | 2054.46 | 2054.07 | SL hit (close<static) qty=1.00 sl=2005.40 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 2010.00 | 2053.54 | 2053.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1988.60 | 2051.64 | 2052.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 09:15:00 | 2073.20 | 2046.07 | 2049.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 2073.20 | 2046.07 | 2049.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 2073.20 | 2046.07 | 2049.37 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 14:15:00 | 2044.50 | 2046.27 | 2049.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 15:15:00 | 2046.20 | 2046.27 | 2049.37 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-07 09:15:00 | 2030.50 | 2046.12 | 2049.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 10:15:00 | 2031.30 | 2045.97 | 2049.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-11 11:15:00 | 2032.10 | 2000.49 | 2015.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:15:00 | 2030.20 | 2000.79 | 2015.10 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-16 13:15:00 | 2042.50 | 2006.77 | 2016.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-16 14:15:00 | 2050.80 | 2007.21 | 2016.92 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-17 13:15:00 | 2043.10 | 2009.51 | 2017.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-17 14:15:00 | 2044.90 | 2009.86 | 2017.94 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-17 15:15:00 | 2041.00 | 2010.17 | 2018.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-18 09:15:00 | 2054.80 | 2010.61 | 2018.24 | ENTRY2 sustain failed after 1080m |
| Cross detected — sustain check pending | 2026-02-19 13:15:00 | 2040.30 | 2015.06 | 2020.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 14:15:00 | 2033.50 | 2015.24 | 2020.18 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-24 09:15:00 | 2032.90 | 2020.49 | 2022.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 2042.00 | 2020.70 | 2022.63 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1994.00 | 2024.15 | 2024.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1966.20 | 2023.58 | 2024.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.53 | 1866.56 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 1775.00 | 1783.40 | 1862.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 1763.00 | 1783.20 | 1862.35 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1769.60 | 1784.33 | 1859.06 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1774.40 | 1784.23 | 1858.64 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1796.77 | 1853.70 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1840.80 | 1798.27 | 1853.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1842.70 | 1798.71 | 1853.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.55 | SL hit (close>ema400) qty=1.00 sl=1853.55 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.55 | SL hit (close>ema400) qty=1.00 sl=1853.55 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.55 | SL hit (close>static) qty=1.00 sl=1854.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 15:15:00 | 1842.90 | 1801.21 | 1853.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1818.00 | 1801.37 | 1853.28 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-15 11:15:00 | 1609.20 | 2024-02-06 09:15:00 | 1610.45 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2023-11-16 10:15:00 | 1612.60 | 2024-02-06 09:15:00 | 1610.45 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2023-11-21 11:15:00 | 1602.30 | 2024-02-06 09:15:00 | 1610.45 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2024-01-24 10:15:00 | 1603.70 | 2024-02-06 09:15:00 | 1610.45 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-02-20 13:15:00 | 1608.85 | 2024-03-26 11:15:00 | 1618.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-02-27 09:15:00 | 1606.05 | 2024-03-26 11:15:00 | 1618.45 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-02-27 13:15:00 | 1601.00 | 2024-03-26 11:15:00 | 1618.45 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-03-01 13:15:00 | 1614.00 | 2024-03-26 11:15:00 | 1618.45 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-03-05 13:15:00 | 1573.30 | 2024-03-28 09:15:00 | 1637.60 | STOP_HIT | 1.00 | -4.09% |
| SELL | retest2 | 2024-03-11 15:15:00 | 1595.00 | 2024-03-28 09:15:00 | 1637.60 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2024-03-21 12:15:00 | 1600.45 | 2024-03-28 09:15:00 | 1637.60 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-03-22 09:15:00 | 1596.20 | 2024-03-28 09:15:00 | 1637.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-03-27 13:15:00 | 1587.00 | 2024-03-28 09:15:00 | 1637.60 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2024-04-25 14:15:00 | 1654.70 | 2024-04-26 09:15:00 | 1604.30 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-05-03 10:15:00 | 1655.95 | 2024-05-03 15:15:00 | 1623.00 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-05-29 11:15:00 | 1575.15 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-06-12 14:15:00 | 1578.95 | 2024-06-25 13:15:00 | 1604.75 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-20 10:15:00 | 1580.50 | 2024-06-28 09:15:00 | 1611.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-06-21 10:15:00 | 1585.00 | 2024-06-28 09:15:00 | 1611.90 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1579.90 | 2024-06-28 09:15:00 | 1611.90 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-06-25 10:15:00 | 1573.55 | 2024-06-28 09:15:00 | 1611.90 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2024-07-01 10:15:00 | 1580.55 | 2024-07-04 09:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-07-01 14:15:00 | 1580.00 | 2024-07-04 09:15:00 | 1599.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-07-08 10:15:00 | 1568.30 | 2024-07-12 09:15:00 | 1593.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-29 15:15:00 | 1617.15 | 2024-08-05 10:15:00 | 1575.30 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest1 | 2024-10-09 13:15:00 | 1877.35 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2024-10-10 10:15:00 | 1881.05 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2024-10-10 13:15:00 | 1879.55 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2024-10-11 11:15:00 | 1876.15 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1825.65 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest1 | 2024-12-18 10:15:00 | 1630.00 | 2025-01-02 12:15:00 | 1711.80 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest2 | 2025-03-05 11:15:00 | 1786.10 | 2025-03-26 09:15:00 | 1964.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1943.50 | 2025-09-04 12:15:00 | 2016.10 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-29 15:15:00 | 2024.70 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-10-06 10:15:00 | 2021.70 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-08 13:15:00 | 2014.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-09 11:15:00 | 2022.00 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-13 15:15:00 | 2023.50 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-10-14 15:15:00 | 2019.40 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-12 13:15:00 | 2034.50 | 2025-12-29 10:15:00 | 2004.20 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-12-17 14:15:00 | 2019.90 | 2025-12-29 10:15:00 | 2004.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-07 10:15:00 | 2031.30 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2030.20 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-19 14:15:00 | 2033.50 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-24 10:15:00 | 2042.00 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-09 13:15:00 | 1763.00 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest1 | 2026-04-13 10:15:00 | 1774.40 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1842.70 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -0.79% |
