# PB Fintech Ltd. (POLICYBZR)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1647.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 38 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 34
- **Target hits / Stop hits / Partials:** 0 / 38 / 4
- **Avg / median % per leg:** -0.78% / -1.54%
- **Sum % (uncompounded):** -32.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 0 | 0.0% | 0 | 34 | 0 | -1.70% | -57.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 34 | 0 | 0.0% | 0 | 34 | 0 | -1.70% | -57.9% |
| SELL (all) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.13% | 25.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 0 | 4 | 4 | 3.13% | 25.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 8 | 19.0% | 0 | 38 | 4 | -0.78% | -32.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 1708.00 | 1613.49 | 1613.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1745.90 | 1614.81 | 1613.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 1638.60 | 1645.95 | 1630.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 1638.60 | 1645.95 | 1630.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1638.60 | 1645.95 | 1630.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 1633.00 | 1645.95 | 1630.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1773.40 | 1824.58 | 1774.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1772.00 | 1824.58 | 1774.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1774.80 | 1824.09 | 1774.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 1789.20 | 1823.75 | 1774.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 13:15:00 | 1802.70 | 1822.05 | 1775.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:45:00 | 1784.50 | 1823.83 | 1782.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:30:00 | 1792.30 | 1822.85 | 1782.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1793.50 | 1822.56 | 1782.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 1796.30 | 1822.56 | 1782.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1797.40 | 1824.64 | 1787.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 1800.00 | 1824.64 | 1787.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1790.90 | 1824.06 | 1787.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:45:00 | 1790.80 | 1824.06 | 1787.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1791.10 | 1823.40 | 1787.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 1750.50 | 1822.02 | 1787.48 | SL hit (close<static) qty=1.00 sl=1768.70 alert=retest2 |

### Cycle 2 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1753.50 | 1807.99 | 1808.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1722.00 | 1807.13 | 1807.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 1778.20 | 1769.41 | 1786.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:00:00 | 1778.20 | 1769.41 | 1786.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1775.30 | 1769.47 | 1786.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:30:00 | 1784.90 | 1769.47 | 1786.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1772.70 | 1769.50 | 1786.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 1762.20 | 1769.43 | 1785.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 1763.20 | 1769.37 | 1785.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 1762.60 | 1769.22 | 1785.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1738.10 | 1769.19 | 1785.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 12:15:00 | 1675.04 | 1757.02 | 1776.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 14:15:00 | 1674.09 | 1755.39 | 1775.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 14:15:00 | 1674.47 | 1755.39 | 1775.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-17 10:15:00 | 1651.19 | 1748.71 | 1771.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 12:15:00 | 1734.40 | 1726.64 | 1755.79 | SL hit (close>ema200) qty=0.50 sl=1726.64 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 12:15:00 | 1847.80 | 1770.63 | 1770.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 1856.00 | 1771.48 | 1770.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1775.60 | 1776.61 | 1773.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1775.60 | 1776.61 | 1773.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1775.60 | 1776.61 | 1773.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 10:30:00 | 1794.60 | 1776.34 | 1773.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:30:00 | 1795.20 | 1777.13 | 1774.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:00:00 | 1795.10 | 1844.22 | 1815.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 1796.50 | 1843.72 | 1815.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 1758.60 | 1841.66 | 1815.14 | SL hit (close<static) qty=1.00 sl=1764.20 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1691.40 | 1812.38 | 1812.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1665.60 | 1810.92 | 1812.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1509.70 | 1505.65 | 1574.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 09:45:00 | 1512.50 | 1505.65 | 1574.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1528.60 | 1482.52 | 1527.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:00:00 | 1528.60 | 1482.52 | 1527.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 1551.00 | 1483.20 | 1527.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 15:00:00 | 1551.00 | 1483.20 | 1527.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 1703.30 | 1560.09 | 1559.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1712.40 | 1586.74 | 1573.73 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-04 13:45:00 | 1789.20 | 2025-07-18 09:15:00 | 1750.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-07-07 13:15:00 | 1802.70 | 2025-07-18 09:15:00 | 1750.50 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2025-07-11 13:45:00 | 1784.50 | 2025-07-18 09:15:00 | 1750.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-07-14 09:30:00 | 1792.30 | 2025-07-18 09:15:00 | 1750.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-07-21 12:45:00 | 1804.90 | 2025-07-24 09:15:00 | 1766.50 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1797.70 | 2025-07-24 09:15:00 | 1766.50 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-24 11:30:00 | 1794.30 | 2025-07-25 09:15:00 | 1778.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-24 12:00:00 | 1795.30 | 2025-07-25 09:15:00 | 1778.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-07-25 11:30:00 | 1779.70 | 2025-07-28 11:15:00 | 1765.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-07-25 14:30:00 | 1780.70 | 2025-07-28 11:15:00 | 1765.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-28 15:00:00 | 1782.70 | 2025-08-01 13:15:00 | 1770.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-07-29 12:00:00 | 1780.30 | 2025-08-01 13:15:00 | 1770.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-07-29 15:15:00 | 1794.00 | 2025-08-01 13:15:00 | 1770.30 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1797.90 | 2025-08-04 09:15:00 | 1767.10 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-08-01 10:45:00 | 1798.50 | 2025-08-04 09:15:00 | 1767.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-08-11 09:15:00 | 1802.10 | 2025-08-29 09:15:00 | 1774.40 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-08-11 11:15:00 | 1817.90 | 2025-08-29 09:15:00 | 1774.40 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-08-11 12:00:00 | 1818.90 | 2025-08-29 09:15:00 | 1774.40 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-08-28 09:30:00 | 1821.80 | 2025-08-29 09:15:00 | 1774.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-08-28 10:15:00 | 1820.10 | 2025-08-29 09:15:00 | 1774.40 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-09-02 11:00:00 | 1831.80 | 2025-09-08 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-09-02 12:15:00 | 1838.20 | 2025-09-08 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1834.50 | 2025-09-08 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-09-03 11:15:00 | 1831.40 | 2025-09-08 09:15:00 | 1801.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-09-16 09:30:00 | 1828.50 | 2025-09-17 12:15:00 | 1799.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-16 15:00:00 | 1821.20 | 2025-09-17 12:15:00 | 1799.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-10-08 11:00:00 | 1762.20 | 2025-10-15 12:15:00 | 1675.04 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-10-08 12:00:00 | 1763.20 | 2025-10-15 14:15:00 | 1674.09 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-10-08 14:30:00 | 1762.60 | 2025-10-15 14:15:00 | 1674.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-09 09:15:00 | 1738.10 | 2025-10-17 10:15:00 | 1651.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 11:00:00 | 1762.20 | 2025-10-27 12:15:00 | 1734.40 | STOP_HIT | 0.50 | 1.58% |
| SELL | retest2 | 2025-10-08 12:00:00 | 1763.20 | 2025-10-27 12:15:00 | 1734.40 | STOP_HIT | 0.50 | 1.63% |
| SELL | retest2 | 2025-10-08 14:30:00 | 1762.60 | 2025-10-27 12:15:00 | 1734.40 | STOP_HIT | 0.50 | 1.60% |
| SELL | retest2 | 2025-10-09 09:15:00 | 1738.10 | 2025-10-27 12:15:00 | 1734.40 | STOP_HIT | 0.50 | 0.21% |
| BUY | retest2 | 2025-11-26 10:30:00 | 1794.60 | 2025-12-17 14:15:00 | 1758.60 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-11-27 09:30:00 | 1795.20 | 2025-12-17 14:15:00 | 1758.60 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-12-17 11:00:00 | 1795.10 | 2025-12-17 14:15:00 | 1758.60 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-12-17 11:30:00 | 1796.50 | 2025-12-17 14:15:00 | 1758.60 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-31 12:30:00 | 1838.40 | 2026-01-01 13:15:00 | 1813.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-31 13:00:00 | 1840.00 | 2026-01-01 13:15:00 | 1813.50 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-01 09:15:00 | 1837.70 | 2026-01-01 13:15:00 | 1813.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2026-01-01 11:15:00 | 1840.80 | 2026-01-01 13:15:00 | 1813.50 | STOP_HIT | 1.00 | -1.48% |
