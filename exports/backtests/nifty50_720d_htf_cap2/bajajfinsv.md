# BAJAJFINSV (BAJAJFINSV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1836.10
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 12 |
| PENDING | 40 |
| PENDING_CANCEL | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 30 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 8
- **Winners / losers:** 10 / 19
- **Target hits / Stop hits / Partials:** 0 / 28 / 1
- **Avg / median % per leg:** 0.51% / -1.04%
- **Sum % (uncompounded):** 14.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 10 | 50.0% | 0 | 19 | 1 | 3.24% | 64.9% |
| BUY @ 2nd Alert (retest1) | 5 | 3 | 60.0% | 0 | 4 | 1 | 3.16% | 15.8% |
| BUY @ 3rd Alert (retest2) | 15 | 7 | 46.7% | 0 | 15 | 0 | 3.27% | 49.1% |
| SELL (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -5.56% | -50.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.21% | -5.2% |
| SELL @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -5.60% | -44.8% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.76% | 10.6% |
| retest2 (combined) | 23 | 7 | 30.4% | 0 | 23 | 0 | 0.19% | 4.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 1543.65 | 1524.99 | 1524.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 11:15:00 | 1549.15 | 1525.24 | 1525.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 09:15:00 | 1525.70 | 1528.26 | 1526.65 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 09:15:00 | 1525.70 | 1528.26 | 1526.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 09:15:00 | 1525.70 | 1528.26 | 1526.65 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-09-25 09:15:00 | 1582.45 | 1529.56 | 1527.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-25 10:15:00 | 1582.10 | 1530.09 | 1527.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-03 12:15:00 | 1553.45 | 1538.48 | 1532.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 13:15:00 | 1554.10 | 1538.63 | 1532.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-10-04 13:15:00 | 1523.65 | 1538.58 | 1533.06 | SL hit qty=1.00 sl=1523.65 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-04 13:15:00 | 1523.65 | 1538.58 | 1533.06 | SL hit qty=1.00 sl=1523.65 alert=retest2 |
| Cross detected — sustain check pending | 2023-10-05 11:15:00 | 1552.40 | 1538.75 | 1533.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2023-10-05 12:15:00 | 1548.25 | 1538.84 | 1533.36 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2023-10-06 09:15:00 | 1576.45 | 1539.35 | 1533.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 10:15:00 | 1594.00 | 1539.89 | 1534.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-30 10:15:00 | 1554.70 | 1588.81 | 1568.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 11:15:00 | 1554.60 | 1588.47 | 1568.28 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 1560.40 | 1588.19 | 1568.24 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2023-10-31 09:15:00 | 1562.05 | 1587.10 | 1568.08 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 10:15:00 | 1568.05 | 1586.91 | 1568.08 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2023-10-31 14:15:00 | 1571.15 | 1586.14 | 1568.06 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 15:15:00 | 1569.55 | 1585.98 | 1568.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-03 13:15:00 | 1552.05 | 1584.07 | 1568.69 | SL hit qty=1.00 sl=1552.05 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-11-03 13:15:00 | 1552.05 | 1584.07 | 1568.69 | SL hit qty=1.00 sl=1552.05 alert=retest2 |
| Cross detected — sustain check pending | 2023-11-06 12:15:00 | 1561.95 | 1581.88 | 1568.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 13:15:00 | 1564.55 | 1581.71 | 1568.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2023-11-07 11:15:00 | 1552.05 | 1580.84 | 1567.92 | SL hit qty=1.00 sl=1552.05 alert=retest2 |
| Cross detected — sustain check pending | 2023-11-07 13:15:00 | 1568.25 | 1580.51 | 1567.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 14:15:00 | 1569.25 | 1580.40 | 1567.88 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2023-11-08 09:15:00 | 1563.10 | 1580.11 | 1567.87 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2024-02-06 12:15:00 | 1590.00 | 1638.50 | 1638.61 | HTF filter: close above htf_sma |

### Cycle 2 — BUY (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 14:15:00 | 1675.65 | 1612.35 | 1612.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 1692.05 | 1613.79 | 1612.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.28 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1628.35 | 1636.34 | 1625.28 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-04-25 13:15:00 | 1668.90 | 1632.30 | 1625.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 14:15:00 | 1654.70 | 1632.53 | 1625.32 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1623.65 | 1632.47 | 1625.36 | SL hit qty=1.00 sl=1623.65 alert=retest2 |
| Cross detected — sustain check pending | 2024-05-03 09:15:00 | 1660.05 | 1626.88 | 1623.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:15:00 | 1655.95 | 1627.17 | 1623.40 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-05-03 12:15:00 | 1623.65 | 1627.20 | 1623.45 | SL hit qty=1.00 sl=1623.65 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 10:15:00 | 1570.90 | 1620.16 | 1620.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 1562.20 | 1617.23 | 1618.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 1607.75 | 1602.89 | 1610.30 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-05-29 10:15:00 | 1579.10 | 1602.35 | 1609.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:15:00 | 1575.15 | 1602.08 | 1608.97 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-12 13:15:00 | 1580.65 | 1574.53 | 1590.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-12 14:15:00 | 1578.95 | 1574.57 | 1590.16 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-13 10:15:00 | 1583.60 | 1574.82 | 1590.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-13 11:15:00 | 1586.00 | 1574.93 | 1590.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-14 10:15:00 | 1585.05 | 1575.72 | 1589.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-06-14 11:15:00 | 1591.85 | 1575.88 | 1589.99 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-06-20 09:15:00 | 1579.65 | 1578.51 | 1590.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1580.50 | 1578.53 | 1590.05 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-21 09:15:00 | 1585.00 | 1578.84 | 1589.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 1585.00 | 1578.90 | 1589.84 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1593.60 | 1579.20 | 1589.82 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-21 14:15:00 | 1572.80 | 1579.13 | 1589.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1579.90 | 1579.14 | 1589.69 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-06-25 09:15:00 | 1579.65 | 1579.24 | 1589.32 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 10:15:00 | 1573.55 | 1579.18 | 1589.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1595.45 | 1579.51 | 1589.26 | SL hit qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 1595.45 | 1579.51 | 1589.26 | SL hit qty=1.00 sl=1595.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1610.80 | 1580.21 | 1589.47 | SL hit qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1610.80 | 1580.21 | 1589.47 | SL hit qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1610.80 | 1580.21 | 1589.47 | SL hit qty=1.00 sl=1610.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 1610.80 | 1580.21 | 1589.47 | SL hit qty=1.00 sl=1610.80 alert=retest2 |
| Cross detected — sustain check pending | 2024-07-01 09:15:00 | 1584.05 | 1584.03 | 1590.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 10:15:00 | 1580.55 | 1583.99 | 1590.51 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2024-07-01 13:15:00 | 1579.95 | 1583.93 | 1590.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 1580.00 | 1583.89 | 1590.33 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2024-07-03 09:15:00 | 1586.20 | 1583.18 | 1589.68 | max_alert3_locks_per_cycle=2 reached — end cycle |

### Cycle 4 — BUY (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 13:15:00 | 1646.75 | 1592.28 | 1592.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1665.00 | 1598.03 | 1595.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 1594.50 | 1606.18 | 1600.19 | EMA400 retest candle locked |

### Cycle 5 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 1561.10 | 1594.97 | 1595.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1545.50 | 1590.34 | 1592.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 10:15:00 | 1581.00 | 1580.28 | 1587.02 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 1590.55 | 1580.38 | 1587.03 | EMA400 retest candle locked |

### Cycle 6 — BUY (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 12:15:00 | 1673.30 | 1592.74 | 1592.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 1679.30 | 1593.60 | 1593.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.78 | 1782.55 | EMA200 retest candle locked |
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
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.25 | 1859.53 | 1800.28 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1800.28 | 1859.53 | 1800.28 | SL hit qty=1.00 sl=1800.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1800.28 | 1859.53 | 1800.28 | SL hit qty=1.00 sl=1800.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1800.28 | 1859.53 | 1800.28 | SL hit qty=1.00 sl=1800.28 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 1800.28 | 1859.53 | 1800.28 | SL hit qty=1.00 sl=1800.28 alert=retest1 |
| Cross detected — sustain check pending | 2024-10-18 10:15:00 | 1821.10 | 1859.15 | 1800.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 1825.65 | 1858.81 | 1800.51 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1799.50 | 1856.79 | 1800.93 | SL hit qty=1.00 sl=1799.50 alert=retest2 |
| CROSSOVER_SKIP | 2024-11-13 10:15:00 | 1679.60 | 1770.94 | 1771.05 | HTF filter: close above htf_sma |

### Cycle 7 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.22 | 1679.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.55 | 1682.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 10:15:00 | 1806.30 | 1815.29 | 1769.94 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1778.20 | 1813.78 | 1770.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1778.20 | 1813.78 | 1770.52 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-03-05 10:15:00 | 1790.45 | 1813.55 | 1770.62 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:15:00 | 1786.10 | 1813.28 | 1770.69 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 09:15:00 | 2054.01 | 1905.63 | 1856.95 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-08-05 14:15:00 | 1940.00 | 1992.63 | 1992.63 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.08 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 8 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.45 | 2006.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 2001.40 | 2025.07 | 2006.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.07 | 2006.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 14:15:00 | 2025.10 | 2023.27 | 2006.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 2024.70 | 2023.29 | 2006.28 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 2000.10 | 2022.02 | 2006.30 | SL hit qty=1.00 sl=2000.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 2024.00 | 2018.85 | 2005.72 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 2021.70 | 2018.88 | 2005.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 2016.50 | 2020.85 | 2007.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 2014.30 | 2020.78 | 2007.87 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 2020.20 | 2020.47 | 2007.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 2022.00 | 2020.48 | 2008.04 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2010.00 | 2020.20 | 2008.38 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2000.10 | 2019.62 | 2008.32 | SL hit qty=1.00 sl=2000.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2000.10 | 2019.62 | 2008.32 | SL hit qty=1.00 sl=2000.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 2000.10 | 2019.62 | 2008.32 | SL hit qty=1.00 sl=2000.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 14:15:00 | 2019.20 | 2019.34 | 2008.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 2023.50 | 2019.38 | 2008.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 14:15:00 | 2019.00 | 2019.32 | 2008.83 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 2019.40 | 2019.32 | 2008.88 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 2005.40 | 2079.53 | 2053.09 | SL hit qty=1.00 sl=2005.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 2005.40 | 2079.53 | 2053.09 | SL hit qty=1.00 sl=2005.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-12 12:15:00 | 2028.50 | 2071.25 | 2050.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:15:00 | 2034.50 | 2070.89 | 2050.04 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 13:15:00 | 2017.30 | 2065.76 | 2058.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 2019.90 | 2065.30 | 2058.71 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| ALERT3_SKIP | 2025-12-17 15:15:00 | 2021.20 | 2064.86 | 2058.52 | max_alert3_locks_per_cycle=2 reached — end cycle |
| CROSSOVER_SKIP | 2025-12-29 12:15:00 | 2010.00 | 2053.54 | 2053.61 | HTF filter: close above htf_sma |

### Cycle 9 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 2050.20 | 2024.62 | 2024.52 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1994.00 | 2024.15 | 2024.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1966.20 | 2023.58 | 2024.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.53 | 1866.56 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 1775.00 | 1783.40 | 1862.84 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 1763.00 | 1783.20 | 1862.35 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1769.60 | 1784.33 | 1859.06 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1774.40 | 1784.23 | 1858.64 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1796.77 | 1853.70 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 1853.70 | 1796.77 | 1853.70 | SL hit qty=1.00 sl=1853.70 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-21 13:15:00 | 1853.70 | 1796.77 | 1853.70 | SL hit qty=1.00 sl=1853.70 alert=retest1 |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1840.80 | 1798.27 | 1853.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1842.70 | 1798.71 | 1853.55 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1854.80 | 1799.80 | 1853.55 | SL hit qty=1.00 sl=1854.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 15:15:00 | 1842.90 | 1801.21 | 1853.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1818.00 | 1801.37 | 1853.28 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-25 10:15:00 | 1582.10 | 2023-10-04 13:15:00 | 1523.65 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2023-10-03 13:15:00 | 1554.10 | 2023-10-04 13:15:00 | 1523.65 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2023-10-06 10:15:00 | 1594.00 | 2023-11-03 13:15:00 | 1552.05 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2023-10-30 11:15:00 | 1554.60 | 2023-11-03 13:15:00 | 1552.05 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2023-10-31 10:15:00 | 1568.05 | 2023-11-07 11:15:00 | 1552.05 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-10-31 15:15:00 | 1569.55 | 2024-04-26 09:15:00 | 1623.65 | STOP_HIT | 1.00 | 3.45% |
| BUY | retest2 | 2023-11-06 13:15:00 | 1564.55 | 2024-05-03 12:15:00 | 1623.65 | STOP_HIT | 1.00 | 3.78% |
| SELL | retest2 | 2024-05-29 11:15:00 | 1575.15 | 2024-06-25 13:15:00 | 1595.45 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-06-12 14:15:00 | 1578.95 | 2024-06-25 13:15:00 | 1595.45 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2024-06-20 10:15:00 | 1580.50 | 2024-06-26 09:15:00 | 1610.80 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-06-21 10:15:00 | 1585.00 | 2024-06-26 09:15:00 | 1610.80 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1579.90 | 2024-06-26 09:15:00 | 1610.80 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-06-25 10:15:00 | 1573.55 | 2024-06-26 09:15:00 | 1610.80 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2023-11-07 14:15:00 | 1569.25 | 2024-10-18 09:15:00 | 1800.28 | STOP_HIT | 1.00 | 14.72% |
| BUY | retest2 | 2024-04-25 14:15:00 | 1654.70 | 2024-10-18 09:15:00 | 1800.28 | STOP_HIT | 1.00 | 8.80% |
| BUY | retest2 | 2024-05-03 10:15:00 | 1655.95 | 2024-10-18 09:15:00 | 1800.28 | STOP_HIT | 1.00 | 8.72% |
| BUY | retest1 | 2024-10-09 13:15:00 | 1877.35 | 2024-10-18 09:15:00 | 1800.28 | STOP_HIT | 1.00 | -4.11% |
| BUY | retest1 | 2024-10-10 10:15:00 | 1881.05 | 2024-10-21 09:15:00 | 1799.50 | STOP_HIT | 1.00 | -4.34% |
| BUY | retest1 | 2024-10-10 13:15:00 | 1879.55 | 2025-04-21 09:15:00 | 2054.01 | PARTIAL | 0.50 | 9.28% |
| BUY | retest1 | 2024-10-10 13:15:00 | 1879.55 | 2025-09-11 11:15:00 | 2036.30 | STOP_HIT | 0.50 | 8.34% |
| BUY | retest1 | 2024-10-11 11:15:00 | 1876.15 | 2025-10-01 09:15:00 | 2000.10 | STOP_HIT | 1.00 | 6.61% |
| BUY | retest2 | 2024-10-18 11:15:00 | 1825.65 | 2025-10-13 09:15:00 | 2000.10 | STOP_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2025-03-05 11:15:00 | 1786.10 | 2025-10-13 09:15:00 | 2000.10 | STOP_HIT | 1.00 | 11.98% |
| BUY | retest2 | 2025-09-29 15:15:00 | 2024.70 | 2025-10-13 09:15:00 | 2000.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-10-06 10:15:00 | 2021.70 | 2025-11-11 09:15:00 | 2005.40 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-08 13:15:00 | 2014.30 | 2025-11-11 09:15:00 | 2005.40 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-07-01 10:15:00 | 1580.55 | 2026-04-21 13:15:00 | 1853.70 | STOP_HIT | 1.00 | -17.28% |
| SELL | retest2 | 2024-07-01 14:15:00 | 1580.00 | 2026-04-21 13:15:00 | 1853.70 | STOP_HIT | 1.00 | -17.32% |
| SELL | retest1 | 2026-04-09 13:15:00 | 1763.00 | 2026-04-22 12:15:00 | 1854.80 | STOP_HIT | 1.00 | -5.21% |
