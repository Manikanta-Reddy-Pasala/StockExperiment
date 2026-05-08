# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2023-06-08 09:15:00 → 2026-05-08 15:30:00 (4997 bars)
- **Last close:** 1818.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15% (ENTRY1) / 15% (ENTRY2), trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT2_SKIP | 10 |
| ALERT3 | 21 |
| PENDING | 64 |
| PENDING_CANCEL | 13 |
| ENTRY1 | 7 |
| ENTRY2 | 44 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 7 / 44
- **Target hits / Stop hits / Partials:** 0 / 50 / 1
- **Avg / median % per leg:** -1.22% / -1.73%
- **Sum % (uncompounded):** -62.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 7 | 29.2% | 0 | 23 | 1 | -0.27% | -6.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.30% | -17.2% |
| BUY @ 3rd Alert (retest2) | 20 | 7 | 35.0% | 0 | 19 | 1 | 0.54% | 10.7% |
| SELL (all) | 27 | 0 | 0.0% | 0 | 27 | 0 | -2.07% | -55.9% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -5.01% | -15.0% |
| SELL @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.70% | -40.9% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -4.60% | -32.2% |
| retest2 (combined) | 44 | 7 | 15.9% | 0 | 43 | 1 | -0.69% | -30.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 09:15:00 | 1554.90 | 1526.82 | 1526.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 09:15:00 | 1582.45 | 1529.62 | 1528.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 1535.65 | 1538.24 | 1533.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 1535.65 | 1538.24 | 1533.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1535.65 | 1538.24 | 1533.23 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-10-03 11:15:00 | 1549.65 | 1538.37 | 1533.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 12:15:00 | 1553.45 | 1538.52 | 1533.45 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 1526.55 | 1538.98 | 1533.78 | SL hit (close<static) qty=1.00 sl=1527.10 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-05 09:15:00 | 1547.10 | 1538.58 | 1533.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:15:00 | 1545.80 | 1538.65 | 1533.81 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-30 10:15:00 | 1554.70 | 1588.83 | 1568.70 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 11:15:00 | 1554.60 | 1588.49 | 1568.63 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-06 09:15:00 | 1542.95 | 1582.71 | 1568.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 10:15:00 | 1545.40 | 1582.33 | 1568.42 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 11:15:00 | 1558.00 | 1582.09 | 1568.37 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2023-11-06 12:15:00 | 1561.95 | 1581.89 | 1568.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 13:15:00 | 1564.55 | 1581.72 | 1568.31 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-11-07 12:15:00 | 1560.00 | 1580.64 | 1568.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:15:00 | 1568.25 | 1580.52 | 1568.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-06 13:15:00 | 1584.80 | 1637.97 | 1638.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 11:15:00 | 1613.20 | 1609.83 | 1622.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 14:15:00 | 1619.10 | 1610.00 | 1622.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 1619.10 | 1610.00 | 1622.03 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-02-20 10:15:00 | 1610.00 | 1610.15 | 1621.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-02-20 11:15:00 | 1616.00 | 1610.21 | 1621.90 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-02-20 12:15:00 | 1609.05 | 1610.20 | 1621.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-20 13:15:00 | 1608.85 | 1610.18 | 1621.77 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-02-26 15:15:00 | 1614.30 | 1608.14 | 1619.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 09:15:00 | 1606.05 | 1608.12 | 1618.94 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Cross detected — sustain check pending | 2024-02-27 12:15:00 | 1613.10 | 1608.35 | 1618.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:15:00 | 1601.00 | 1608.28 | 1618.80 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-01 10:15:00 | 1612.10 | 1606.18 | 1616.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-03-01 11:15:00 | 1615.30 | 1606.27 | 1616.79 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-03-01 12:15:00 | 1611.30 | 1606.32 | 1616.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 13:15:00 | 1614.00 | 1606.39 | 1616.75 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 1605.75 | 1606.51 | 1616.65 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-03-05 12:15:00 | 1598.40 | 1606.99 | 1616.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-05 13:15:00 | 1573.30 | 1606.65 | 1616.19 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-11 14:15:00 | 1599.60 | 1600.40 | 1611.82 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 15:15:00 | 1595.00 | 1600.34 | 1611.73 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-21 11:15:00 | 1595.45 | 1591.11 | 1603.91 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:15:00 | 1600.45 | 1591.20 | 1603.89 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-03-21 15:15:00 | 1602.60 | 1591.56 | 1603.89 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 09:15:00 | 1596.20 | 1591.61 | 1603.85 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1611.05 | 1591.74 | 1603.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.62 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.62 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.62 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 11:15:00 | 1618.45 | 1592.23 | 1603.62 | SL hit (close>static) qty=1.00 sl=1618.20 alert=retest2 |
| Cross detected — sustain check pending | 2024-03-27 12:15:00 | 1585.00 | 1592.69 | 1603.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-27 13:15:00 | 1587.00 | 1592.64 | 1603.33 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.29 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.29 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.29 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.29 | SL hit (close>static) qty=1.00 sl=1625.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 1637.60 | 1592.87 | 1603.29 | SL hit (close>static) qty=1.00 sl=1615.60 alert=retest2 |

### Cycle 3 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1675.65 | 1612.35 | 1612.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1692.05 | 1613.79 | 1612.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.29 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-04-25 13:15:00 | 1668.90 | 1632.30 | 1625.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:15:00 | 1654.70 | 1632.53 | 1625.32 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1604.30 | 1632.47 | 1625.37 | SL hit (close<static) qty=1.00 sl=1623.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-03 09:15:00 | 1660.05 | 1626.88 | 1623.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1655.95 | 1627.17 | 1623.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-03 15:15:00 | 1623.00 | 1627.15 | 1623.49 | SL hit (close<static) qty=1.00 sl=1623.65 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1570.90 | 1620.16 | 1620.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1562.20 | 1617.23 | 1618.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-05-29 10:15:00 | 1579.10 | 1602.35 | 1609.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 1575.15 | 1602.08 | 1608.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-12 13:15:00 | 1580.65 | 1574.53 | 1590.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 14:15:00 | 1578.95 | 1574.57 | 1590.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-13 10:15:00 | 1583.60 | 1574.82 | 1590.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-13 11:15:00 | 1586.00 | 1574.93 | 1590.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 10:15:00 | 1585.05 | 1575.72 | 1589.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 11:15:00 | 1591.85 | 1575.88 | 1590.00 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-20 09:15:00 | 1579.65 | 1578.51 | 1590.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1580.50 | 1578.53 | 1590.05 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 09:15:00 | 1585.00 | 1578.84 | 1589.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1585.00 | 1578.90 | 1589.84 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1593.60 | 1579.20 | 1589.82 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2024-06-21 14:15:00 | 1572.80 | 1579.13 | 1589.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1579.90 | 1579.14 | 1589.69 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-25 09:15:00 | 1579.65 | 1579.24 | 1589.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1573.55 | 1579.18 | 1589.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1604.75 | 1579.51 | 1589.26 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1604.75 | 1579.51 | 1589.26 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.41 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.41 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.41 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 1611.90 | 1583.25 | 1590.41 | SL hit (close>static) qty=1.00 sl=1610.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-01 09:15:00 | 1584.05 | 1584.03 | 1590.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 1580.55 | 1583.99 | 1590.51 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-01 13:15:00 | 1579.95 | 1583.93 | 1590.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 1580.00 | 1583.89 | 1590.33 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 1586.20 | 1583.18 | 1589.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 1599.00 | 1583.77 | 1589.76 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 1599.00 | 1583.77 | 1589.76 | SL hit (close>static) qty=1.00 sl=1595.45 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-05 13:15:00 | 1572.80 | 1583.81 | 1589.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-05 14:15:00 | 1580.20 | 1583.77 | 1589.42 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-07-08 09:15:00 | 1569.10 | 1583.57 | 1589.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:15:00 | 1568.30 | 1583.42 | 1589.16 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-10 13:15:00 | 1575.00 | 1582.38 | 1588.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-07-10 14:15:00 | 1581.95 | 1582.38 | 1588.11 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 1593.40 | 1582.79 | 1588.07 | SL hit (close>static) qty=1.00 sl=1592.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 1646.75 | 1592.28 | 1592.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1665.00 | 1598.03 | 1595.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1561.10 | 1594.97 | 1595.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.34 | 1592.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.28 | 1587.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.03 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.74 | 1592.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.30 | 1593.60 | 1593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.78 | 1782.55 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-09 10:15:00 | 1877.30 | 1860.36 | 1784.27 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-09 11:15:00 | 1862.00 | 1860.37 | 1784.65 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-09 12:15:00 | 1871.70 | 1860.48 | 1785.09 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 13:15:00 | 1877.35 | 1860.65 | 1785.55 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 09:15:00 | 1879.55 | 1860.95 | 1786.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1881.05 | 1861.15 | 1787.29 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 12:15:00 | 1871.15 | 1861.31 | 1788.10 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 13:15:00 | 1879.55 | 1861.49 | 1788.56 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-11 10:15:00 | 1875.35 | 1861.96 | 1790.24 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1876.15 | 1862.10 | 1790.67 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.25 | 1859.53 | 1800.28 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2024-10-18 10:15:00 | 1821.10 | 1859.15 | 1800.38 | ENTRY2 cross detected — sustain check pending (15m) |
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
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 09:15:00 | 2054.01 | 1905.63 | 1856.95 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1937.90 | 1969.87 | 1903.87 | SL hit (close<ema200) qty=0.50 sl=1969.87 alert=retest2 |

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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 2021.20 | 2064.86 | 2058.52 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-18 09:15:00 | 2023.90 | 2064.45 | 2058.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:15:00 | 2033.60 | 2064.15 | 2058.22 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 2014.60 | 2056.60 | 2055.12 | SL hit (close<static) qty=1.00 sl=2018.00 alert=retest2 |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 2044.10 | 2020.94 | 2022.73 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-24 15:15:00 | 2038.20 | 2021.74 | 2023.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-25 09:15:00 | 2071.40 | 2022.23 | 2023.34 | ENTRY2 sustain failed after 1080m |
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
| BUY | retest2 | 2023-10-03 12:15:00 | 1553.45 | 2023-10-04 09:15:00 | 1526.55 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2023-10-05 10:15:00 | 1545.80 | 2024-02-06 12:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.86% |
| BUY | retest2 | 2023-10-30 11:15:00 | 1554.60 | 2024-02-06 12:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.28% |
| BUY | retest2 | 2023-11-06 10:15:00 | 1545.40 | 2024-02-06 12:15:00 | 1590.00 | STOP_HIT | 1.00 | 2.89% |
| BUY | retest2 | 2023-11-06 13:15:00 | 1564.55 | 2024-02-06 12:15:00 | 1590.00 | STOP_HIT | 1.00 | 1.63% |
| BUY | retest2 | 2023-11-07 13:15:00 | 1568.25 | 2024-02-06 12:15:00 | 1590.00 | STOP_HIT | 1.00 | 1.39% |
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
| BUY | retest1 | 2024-10-09 13:15:00 | 1877.35 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2024-10-10 10:15:00 | 1881.05 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2024-10-10 13:15:00 | 1879.55 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2024-10-11 11:15:00 | 1876.15 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1825.65 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest1 | 2024-12-18 10:15:00 | 1630.00 | 2025-01-02 12:15:00 | 1711.80 | STOP_HIT | 1.00 | -5.02% |
| BUY | retest2 | 2025-03-05 11:15:00 | 1786.10 | 2025-04-21 09:15:00 | 2054.01 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-03-05 11:15:00 | 1786.10 | 2025-04-30 09:15:00 | 1937.90 | STOP_HIT | 0.50 | 8.50% |
| SELL | retest2 | 2025-08-26 10:15:00 | 1943.50 | 2025-09-04 12:15:00 | 2016.10 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-29 15:15:00 | 2024.70 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-10-06 10:15:00 | 2021.70 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-08 13:15:00 | 2014.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-09 11:15:00 | 2022.00 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-13 15:15:00 | 2023.50 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-10-14 15:15:00 | 2019.40 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-12 13:15:00 | 2034.50 | 2025-12-26 12:15:00 | 2014.60 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-17 14:15:00 | 2019.90 | 2025-12-29 10:15:00 | 2004.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-18 10:15:00 | 2033.60 | 2025-12-29 10:15:00 | 2004.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-07 10:15:00 | 2031.30 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-02-11 12:15:00 | 2030.20 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-19 14:15:00 | 2033.50 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-02-24 10:15:00 | 2042.00 | 2026-02-26 10:15:00 | 2050.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2026-04-09 13:15:00 | 1763.00 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest1 | 2026-04-13 10:15:00 | 1774.40 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1842.70 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -0.79% |
