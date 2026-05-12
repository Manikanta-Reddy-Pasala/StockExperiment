# Tata Communications Ltd. (TATACOMM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1582.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 0 |
| ALERT3 | 34 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 5 |
| TARGET_HIT | 5 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 13
- **Target hits / Stop hits / Partials:** 5 / 13 / 5
- **Avg / median % per leg:** 2.28% / -1.41%
- **Sum % (uncompounded):** 52.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.78% | -21.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 0 | 0.0% | 0 | 12 | 0 | -1.78% | -21.3% |
| SELL (all) | 11 | 10 | 90.9% | 5 | 1 | 5 | 6.70% | 73.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 10 | 90.9% | 5 | 1 | 5 | 6.70% | 73.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 10 | 43.5% | 5 | 13 | 5 | 2.28% | 52.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 1644.00 | 1572.55 | 1572.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 10:15:00 | 1664.30 | 1579.42 | 1575.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 09:15:00 | 1665.60 | 1672.23 | 1638.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 10:00:00 | 1665.60 | 1672.23 | 1638.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1645.00 | 1671.66 | 1639.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1645.00 | 1671.66 | 1639.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1632.10 | 1669.86 | 1639.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1632.10 | 1669.86 | 1639.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1651.20 | 1669.67 | 1639.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:30:00 | 1640.90 | 1669.67 | 1639.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1646.60 | 1669.10 | 1639.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1643.70 | 1669.10 | 1639.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1647.50 | 1668.25 | 1639.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 1644.10 | 1668.25 | 1639.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1643.50 | 1668.00 | 1639.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 1640.90 | 1668.00 | 1639.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1652.70 | 1667.63 | 1639.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1654.60 | 1667.63 | 1639.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1691.30 | 1725.63 | 1698.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1691.30 | 1725.63 | 1698.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1670.80 | 1725.09 | 1698.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 1670.80 | 1725.09 | 1698.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1678.90 | 1715.19 | 1695.54 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 1591.30 | 1682.11 | 1682.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1570.60 | 1670.26 | 1676.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1616.50 | 1611.09 | 1637.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1616.50 | 1611.09 | 1637.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1638.30 | 1611.36 | 1637.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 1644.90 | 1611.36 | 1637.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 1652.60 | 1611.77 | 1637.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 1652.60 | 1611.77 | 1637.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 1660.00 | 1612.25 | 1637.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:45:00 | 1661.20 | 1612.25 | 1637.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 1647.10 | 1637.50 | 1646.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 1649.20 | 1637.50 | 1646.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1654.90 | 1637.67 | 1646.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1654.90 | 1637.67 | 1646.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1655.30 | 1637.85 | 1646.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:45:00 | 1656.80 | 1637.85 | 1646.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1651.50 | 1639.59 | 1647.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 15:15:00 | 1645.80 | 1640.37 | 1647.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 1667.00 | 1634.49 | 1642.80 | SL hit (close>static) qty=1.00 sl=1662.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1868.90 | 1651.23 | 1650.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 1888.10 | 1653.59 | 1651.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1823.00 | 1824.58 | 1768.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:00:00 | 1823.00 | 1824.58 | 1768.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1814.90 | 1847.30 | 1811.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 1816.60 | 1847.30 | 1811.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 1809.50 | 1846.93 | 1811.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:45:00 | 1817.40 | 1846.93 | 1811.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 1800.00 | 1846.46 | 1811.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 1790.10 | 1845.90 | 1811.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1803.00 | 1845.47 | 1811.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:30:00 | 1808.00 | 1845.11 | 1811.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 13:30:00 | 1820.70 | 1844.36 | 1811.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1804.50 | 1843.57 | 1811.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 10:00:00 | 1805.20 | 1843.19 | 1811.23 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 1793.80 | 1842.70 | 1811.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 1795.40 | 1842.70 | 1811.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1779.00 | 1839.94 | 1810.68 | SL hit (close<static) qty=1.00 sl=1790.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 13:15:00 | 1725.90 | 1801.08 | 1801.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1705.90 | 1782.54 | 1791.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 14:15:00 | 1678.00 | 1647.55 | 1703.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 1678.00 | 1647.55 | 1703.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1670.30 | 1648.17 | 1702.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 10:30:00 | 1662.10 | 1648.38 | 1702.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:30:00 | 1663.60 | 1653.19 | 1701.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 1652.00 | 1653.93 | 1700.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:30:00 | 1662.30 | 1653.44 | 1692.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1678.00 | 1654.69 | 1692.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1645.50 | 1654.69 | 1692.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1578.99 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1580.42 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1569.40 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1579.18 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1563.22 | 1645.29 | 1682.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 09:15:00 | 1495.89 | 1638.81 | 1677.45 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-25 15:15:00 | 1645.80 | 2025-10-07 09:15:00 | 1667.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-12-09 11:30:00 | 1808.00 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-12-09 13:30:00 | 1820.70 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1804.50 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-12-10 10:00:00 | 1805.20 | 2025-12-11 09:15:00 | 1779.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-16 11:30:00 | 1838.90 | 2025-12-18 09:15:00 | 1798.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-12-17 09:15:00 | 1838.00 | 2025-12-18 09:15:00 | 1798.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-19 14:30:00 | 1833.50 | 2025-12-22 13:15:00 | 1800.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-12-22 09:45:00 | 1833.00 | 2025-12-22 13:15:00 | 1800.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-30 10:15:00 | 1802.20 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-01-02 10:45:00 | 1804.80 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-02 13:00:00 | 1800.00 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-02 13:30:00 | 1802.80 | 2026-01-08 11:15:00 | 1772.60 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-11 10:30:00 | 1662.10 | 2026-03-02 09:15:00 | 1578.99 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:30:00 | 1663.60 | 2026-03-02 09:15:00 | 1580.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1652.00 | 2026-03-02 09:15:00 | 1569.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1662.30 | 2026-03-02 09:15:00 | 1579.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1645.50 | 2026-03-02 09:15:00 | 1563.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 10:30:00 | 1662.10 | 2026-03-04 09:15:00 | 1495.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 09:30:00 | 1663.60 | 2026-03-04 09:15:00 | 1497.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 15:00:00 | 1652.00 | 2026-03-04 09:15:00 | 1496.07 | TARGET_HIT | 0.50 | 9.44% |
| SELL | retest2 | 2026-02-23 09:30:00 | 1662.30 | 2026-03-04 11:15:00 | 1486.80 | TARGET_HIT | 0.50 | 10.56% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1645.50 | 2026-03-04 14:15:00 | 1480.95 | TARGET_HIT | 0.50 | 10.00% |
