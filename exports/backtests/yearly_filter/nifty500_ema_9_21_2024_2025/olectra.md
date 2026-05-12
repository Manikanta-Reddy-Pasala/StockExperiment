# Olectra Greentech Ltd. (OLECTRA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1345.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 93 |
| ALERT2 | 90 |
| ALERT2_SKIP | 43 |
| ALERT3 | 226 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 105 |
| PARTIAL | 18 |
| TARGET_HIT | 16 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 61
- **Target hits / Stop hits / Partials:** 16 / 89 / 18
- **Avg / median % per leg:** 1.47% / 0.05%
- **Sum % (uncompounded):** 180.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 46 | 21 | 45.7% | 11 | 35 | 0 | 1.60% | 73.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 46 | 21 | 45.7% | 11 | 35 | 0 | 1.60% | 73.8% |
| SELL (all) | 77 | 41 | 53.2% | 5 | 54 | 18 | 1.39% | 107.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 77 | 41 | 53.2% | 5 | 54 | 18 | 1.39% | 107.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 123 | 62 | 50.4% | 16 | 89 | 18 | 1.47% | 180.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 1644.50 | 1625.79 | 1623.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 1653.10 | 1633.31 | 1628.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 1690.10 | 1690.55 | 1671.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:30:00 | 1685.80 | 1690.55 | 1671.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1717.90 | 1707.75 | 1694.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 1745.00 | 1720.27 | 1707.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 1794.00 | 1803.37 | 1803.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 1794.00 | 1803.37 | 1803.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 1785.65 | 1799.83 | 1801.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 13:15:00 | 1812.55 | 1789.11 | 1793.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 13:15:00 | 1812.55 | 1789.11 | 1793.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1812.55 | 1789.11 | 1793.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:45:00 | 1808.65 | 1789.11 | 1793.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1804.00 | 1792.09 | 1794.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 10:00:00 | 1781.00 | 1791.77 | 1794.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 11:00:00 | 1782.20 | 1789.86 | 1793.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 11:15:00 | 1799.00 | 1774.67 | 1772.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 1799.00 | 1774.67 | 1772.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 12:15:00 | 1805.70 | 1780.88 | 1775.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1742.75 | 1783.16 | 1779.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1742.75 | 1783.16 | 1779.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1742.75 | 1783.16 | 1779.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1731.35 | 1783.16 | 1779.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 1662.45 | 1759.02 | 1769.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 1511.95 | 1709.61 | 1745.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 15:15:00 | 1640.00 | 1637.51 | 1671.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:15:00 | 1717.35 | 1637.51 | 1671.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1713.75 | 1652.76 | 1675.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1718.25 | 1652.76 | 1675.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1716.20 | 1665.45 | 1679.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1715.20 | 1665.45 | 1679.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 1697.25 | 1684.82 | 1684.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1716.70 | 1691.20 | 1687.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 09:15:00 | 1771.30 | 1775.45 | 1763.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 09:45:00 | 1773.00 | 1775.45 | 1763.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 1762.80 | 1772.92 | 1763.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 1762.80 | 1772.92 | 1763.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 1751.05 | 1768.54 | 1762.62 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 15:15:00 | 1751.85 | 1758.12 | 1758.87 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 13:15:00 | 1761.95 | 1759.17 | 1759.02 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 15:15:00 | 1752.00 | 1757.56 | 1758.30 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 09:15:00 | 1768.75 | 1759.80 | 1759.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 15:15:00 | 1800.00 | 1781.08 | 1771.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 1775.90 | 1784.47 | 1777.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 14:15:00 | 1775.90 | 1784.47 | 1777.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 14:15:00 | 1775.90 | 1784.47 | 1777.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 15:00:00 | 1775.90 | 1784.47 | 1777.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1777.00 | 1782.98 | 1777.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 1771.60 | 1782.98 | 1777.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1763.05 | 1778.99 | 1776.54 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 1765.00 | 1774.23 | 1774.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 1760.00 | 1771.39 | 1773.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 11:15:00 | 1765.10 | 1764.32 | 1768.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 11:15:00 | 1765.10 | 1764.32 | 1768.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 1765.10 | 1764.32 | 1768.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:00:00 | 1765.10 | 1764.32 | 1768.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1773.15 | 1764.46 | 1767.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 1773.15 | 1764.46 | 1767.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1769.00 | 1765.37 | 1767.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 1766.00 | 1765.37 | 1767.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1754.00 | 1763.09 | 1766.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 14:15:00 | 1744.65 | 1755.06 | 1760.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 09:15:00 | 1818.10 | 1762.97 | 1762.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1818.10 | 1762.97 | 1762.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 10:15:00 | 1831.00 | 1776.58 | 1768.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 14:15:00 | 1808.90 | 1809.71 | 1790.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 14:45:00 | 1812.00 | 1809.71 | 1790.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 1818.00 | 1818.07 | 1806.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 1794.00 | 1818.07 | 1806.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1806.45 | 1815.74 | 1806.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 1801.85 | 1815.74 | 1806.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1793.60 | 1811.31 | 1805.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 1798.70 | 1811.31 | 1805.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1800.95 | 1809.24 | 1804.95 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 1770.00 | 1798.98 | 1800.88 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1817.95 | 1795.42 | 1794.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 13:15:00 | 1853.60 | 1807.05 | 1799.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 1815.10 | 1823.18 | 1812.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 12:00:00 | 1815.10 | 1823.18 | 1812.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1808.00 | 1820.15 | 1811.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 1808.00 | 1820.15 | 1811.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1808.05 | 1817.73 | 1811.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 1818.55 | 1812.96 | 1810.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 1822.35 | 1810.26 | 1810.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 11:15:00 | 1805.50 | 1810.06 | 1810.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 1805.50 | 1810.06 | 1810.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 1803.55 | 1808.11 | 1809.18 | Break + close below crossover candle low |

### Cycle 15 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 1838.15 | 1808.25 | 1807.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 1921.95 | 1837.04 | 1821.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 14:15:00 | 1904.45 | 1904.94 | 1881.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 15:00:00 | 1904.45 | 1904.94 | 1881.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1859.00 | 1895.92 | 1881.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1859.00 | 1895.92 | 1881.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1849.60 | 1886.66 | 1878.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1838.30 | 1886.66 | 1878.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 1854.90 | 1872.15 | 1873.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 14:15:00 | 1849.00 | 1867.52 | 1871.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 15:15:00 | 1815.70 | 1814.75 | 1825.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:15:00 | 1813.30 | 1814.75 | 1825.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 1831.95 | 1818.19 | 1825.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 1831.95 | 1818.19 | 1825.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 1822.10 | 1818.97 | 1825.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 11:45:00 | 1815.60 | 1818.74 | 1824.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 09:15:00 | 1724.82 | 1767.96 | 1790.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 1767.50 | 1752.07 | 1768.20 | SL hit (close>ema200) qty=0.50 sl=1752.07 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 1737.90 | 1721.08 | 1719.21 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 1711.90 | 1718.52 | 1719.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 1708.25 | 1716.47 | 1718.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 11:15:00 | 1583.00 | 1580.63 | 1605.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 12:00:00 | 1583.00 | 1580.63 | 1605.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 1571.65 | 1568.72 | 1579.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:30:00 | 1582.00 | 1568.72 | 1579.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 1723.90 | 1581.98 | 1573.69 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 14:15:00 | 1611.85 | 1622.93 | 1623.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1608.00 | 1614.72 | 1618.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 15:15:00 | 1603.00 | 1598.68 | 1603.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 15:15:00 | 1603.00 | 1598.68 | 1603.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 1603.00 | 1598.68 | 1603.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:15:00 | 1622.35 | 1598.68 | 1603.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 1619.25 | 1602.79 | 1605.38 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 1618.00 | 1609.15 | 1608.03 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 1602.60 | 1607.87 | 1608.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 14:15:00 | 1597.05 | 1602.70 | 1605.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 15:15:00 | 1588.05 | 1575.13 | 1582.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 15:15:00 | 1588.05 | 1575.13 | 1582.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 1588.05 | 1575.13 | 1582.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 1618.70 | 1575.13 | 1582.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1607.00 | 1581.51 | 1584.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:30:00 | 1631.75 | 1581.51 | 1584.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 1599.00 | 1585.00 | 1585.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 10:30:00 | 1599.95 | 1585.00 | 1585.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 1593.95 | 1586.79 | 1586.63 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 1579.15 | 1585.49 | 1586.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 1569.60 | 1581.78 | 1584.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 10:15:00 | 1556.90 | 1555.42 | 1561.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 11:00:00 | 1556.90 | 1555.42 | 1561.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 1559.60 | 1556.26 | 1561.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:45:00 | 1560.25 | 1556.26 | 1561.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 12:15:00 | 1558.25 | 1556.66 | 1561.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 12:45:00 | 1563.00 | 1556.66 | 1561.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 1573.00 | 1559.93 | 1562.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:00:00 | 1573.00 | 1559.93 | 1562.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1573.40 | 1562.62 | 1563.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1573.40 | 1562.62 | 1563.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 15:15:00 | 1567.70 | 1563.64 | 1563.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-04 09:15:00 | 1579.50 | 1566.81 | 1565.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 1561.90 | 1565.83 | 1564.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 10:15:00 | 1561.90 | 1565.83 | 1564.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 10:15:00 | 1561.90 | 1565.83 | 1564.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 11:00:00 | 1561.90 | 1565.83 | 1564.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 11:15:00 | 1563.05 | 1565.27 | 1564.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 1565.00 | 1564.68 | 1564.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:15:00 | 1568.00 | 1564.68 | 1564.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 14:15:00 | 1560.55 | 1563.83 | 1564.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 14:15:00 | 1560.55 | 1563.83 | 1564.06 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 1571.55 | 1565.54 | 1564.80 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1548.90 | 1561.85 | 1563.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1530.00 | 1547.52 | 1554.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 1540.00 | 1536.70 | 1544.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 1570.00 | 1536.70 | 1544.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1556.00 | 1540.56 | 1545.91 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 1561.00 | 1550.53 | 1549.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 13:15:00 | 1603.00 | 1563.83 | 1556.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 1622.20 | 1629.12 | 1613.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 1622.20 | 1629.12 | 1613.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1606.05 | 1624.15 | 1614.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 1606.05 | 1624.15 | 1614.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1610.00 | 1621.32 | 1613.91 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 1603.00 | 1610.12 | 1610.51 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1632.00 | 1614.50 | 1612.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 10:15:00 | 1645.90 | 1620.78 | 1615.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 15:15:00 | 1683.00 | 1685.59 | 1664.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 09:15:00 | 1697.55 | 1685.59 | 1664.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1663.00 | 1681.07 | 1664.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 1663.00 | 1681.07 | 1664.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1674.80 | 1679.82 | 1665.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:15:00 | 1655.00 | 1679.82 | 1665.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1651.45 | 1674.14 | 1664.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 12:00:00 | 1651.45 | 1674.14 | 1664.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 1649.40 | 1669.20 | 1662.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 1668.05 | 1663.07 | 1660.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 1668.35 | 1662.26 | 1661.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:45:00 | 1676.15 | 1665.38 | 1662.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 14:15:00 | 1666.00 | 1665.57 | 1663.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 1670.45 | 1666.54 | 1663.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:30:00 | 1665.00 | 1666.54 | 1663.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1666.00 | 1667.47 | 1664.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 09:30:00 | 1665.45 | 1667.47 | 1664.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 1700.75 | 1674.12 | 1668.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:15:00 | 1723.00 | 1693.93 | 1681.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 12:15:00 | 1672.30 | 1683.77 | 1684.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 1672.30 | 1683.77 | 1684.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 1667.30 | 1676.47 | 1680.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 1666.45 | 1665.35 | 1671.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 10:00:00 | 1666.45 | 1665.35 | 1671.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1663.00 | 1664.88 | 1670.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 11:45:00 | 1654.00 | 1662.72 | 1669.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 10:30:00 | 1656.90 | 1648.68 | 1658.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 09:15:00 | 1688.15 | 1657.24 | 1657.98 | SL hit (close>static) qty=1.00 sl=1673.50 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 10:15:00 | 1717.70 | 1669.33 | 1663.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 12:15:00 | 1736.95 | 1690.95 | 1674.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 10:15:00 | 1717.00 | 1722.15 | 1699.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 1717.00 | 1722.15 | 1699.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 1699.45 | 1715.29 | 1701.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 1702.65 | 1715.29 | 1701.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 1708.35 | 1713.90 | 1702.37 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 1670.85 | 1692.09 | 1694.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 1661.95 | 1686.06 | 1691.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1635.90 | 1600.75 | 1627.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 1635.90 | 1600.75 | 1627.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1635.90 | 1600.75 | 1627.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 1635.90 | 1600.75 | 1627.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 1648.05 | 1610.21 | 1629.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 1657.00 | 1610.21 | 1629.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1659.00 | 1635.23 | 1636.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1691.30 | 1635.23 | 1636.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 1703.00 | 1648.79 | 1642.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 11:15:00 | 1726.65 | 1703.94 | 1687.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1719.05 | 1727.87 | 1708.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 10:00:00 | 1719.05 | 1727.87 | 1708.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1704.55 | 1723.21 | 1707.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 11:00:00 | 1704.55 | 1723.21 | 1707.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1740.05 | 1726.58 | 1710.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 13:15:00 | 1751.10 | 1730.60 | 1713.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 14:00:00 | 1747.95 | 1734.07 | 1717.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 1750.65 | 1737.52 | 1721.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 1710.50 | 1719.99 | 1721.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 1710.50 | 1719.99 | 1721.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 1706.00 | 1717.19 | 1719.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 1692.85 | 1681.20 | 1691.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 1692.85 | 1681.20 | 1691.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 1692.85 | 1681.20 | 1691.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:30:00 | 1693.00 | 1681.20 | 1691.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1682.00 | 1681.36 | 1690.40 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2024-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 13:15:00 | 1724.05 | 1699.88 | 1696.82 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 1675.00 | 1697.02 | 1697.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 12:15:00 | 1667.45 | 1691.10 | 1694.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 15:15:00 | 1647.10 | 1646.03 | 1661.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 09:15:00 | 1642.00 | 1646.03 | 1661.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1664.35 | 1649.69 | 1661.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:00:00 | 1664.35 | 1649.69 | 1661.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 1675.55 | 1654.86 | 1663.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 10:30:00 | 1696.10 | 1654.86 | 1663.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1668.00 | 1661.97 | 1664.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 1672.15 | 1661.97 | 1664.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 14:15:00 | 1646.65 | 1658.91 | 1663.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 1640.00 | 1649.67 | 1658.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:30:00 | 1640.85 | 1627.63 | 1627.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 1646.10 | 1631.33 | 1629.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1646.10 | 1631.33 | 1629.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1657.00 | 1644.72 | 1639.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1614.75 | 1639.25 | 1637.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1614.75 | 1639.25 | 1637.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1614.75 | 1639.25 | 1637.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1614.75 | 1639.25 | 1637.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 1622.00 | 1635.80 | 1636.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 1611.85 | 1620.99 | 1626.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1620.35 | 1619.34 | 1625.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 13:30:00 | 1616.10 | 1619.34 | 1625.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1631.65 | 1620.01 | 1623.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 1632.25 | 1620.01 | 1623.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 1627.20 | 1621.45 | 1624.09 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 1639.40 | 1628.13 | 1626.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 1642.90 | 1631.08 | 1628.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1632.00 | 1633.51 | 1630.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 1632.00 | 1633.51 | 1630.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1632.00 | 1633.51 | 1630.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1626.90 | 1633.51 | 1630.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1628.90 | 1632.58 | 1630.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1628.90 | 1632.58 | 1630.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1630.00 | 1632.07 | 1630.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 13:15:00 | 1628.00 | 1632.07 | 1630.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1627.95 | 1631.24 | 1629.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 14:00:00 | 1627.95 | 1631.24 | 1629.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 14:15:00 | 1612.35 | 1627.47 | 1628.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1605.85 | 1617.94 | 1623.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 12:15:00 | 1431.95 | 1424.22 | 1450.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:45:00 | 1431.55 | 1424.22 | 1450.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1458.75 | 1431.44 | 1445.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1458.75 | 1431.44 | 1445.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1446.80 | 1434.51 | 1445.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1448.00 | 1434.51 | 1445.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1447.30 | 1437.07 | 1445.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 12:30:00 | 1439.95 | 1438.22 | 1445.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 11:15:00 | 1436.80 | 1420.33 | 1420.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 1436.80 | 1420.33 | 1420.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 12:15:00 | 1447.70 | 1425.81 | 1422.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 14:15:00 | 1491.00 | 1491.34 | 1467.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 15:00:00 | 1491.00 | 1491.34 | 1467.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1568.00 | 1577.85 | 1570.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:45:00 | 1567.30 | 1577.85 | 1570.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1565.95 | 1575.47 | 1569.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 1565.95 | 1575.47 | 1569.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1569.00 | 1574.17 | 1569.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1577.65 | 1574.17 | 1569.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:00:00 | 1572.60 | 1582.96 | 1581.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 11:15:00 | 1573.50 | 1580.09 | 1579.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 11:15:00 | 1578.40 | 1579.75 | 1579.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 11:15:00 | 1578.40 | 1579.75 | 1579.81 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 1591.10 | 1578.62 | 1578.54 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 13:15:00 | 1574.10 | 1579.20 | 1579.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-09 14:15:00 | 1572.30 | 1577.82 | 1578.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 12:15:00 | 1579.35 | 1573.27 | 1575.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 12:15:00 | 1579.35 | 1573.27 | 1575.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 1579.35 | 1573.27 | 1575.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:00:00 | 1579.35 | 1573.27 | 1575.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 13:15:00 | 1618.25 | 1582.27 | 1579.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 1649.80 | 1595.77 | 1585.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 1620.45 | 1629.69 | 1615.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:30:00 | 1618.75 | 1629.69 | 1615.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1609.80 | 1625.72 | 1614.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 1609.80 | 1625.72 | 1614.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1602.60 | 1621.09 | 1613.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1602.60 | 1621.09 | 1613.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 1588.30 | 1608.25 | 1609.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 1583.00 | 1589.67 | 1596.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1479.95 | 1479.77 | 1495.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 09:45:00 | 1483.85 | 1479.77 | 1495.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 1455.35 | 1441.26 | 1450.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 1459.80 | 1441.26 | 1450.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 1448.80 | 1442.77 | 1450.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:45:00 | 1456.50 | 1442.77 | 1450.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 12:15:00 | 1450.55 | 1444.33 | 1450.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:45:00 | 1446.30 | 1444.33 | 1450.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 1452.35 | 1445.93 | 1450.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 14:00:00 | 1452.35 | 1445.93 | 1450.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 1447.45 | 1446.24 | 1450.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1432.15 | 1447.78 | 1450.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 10:00:00 | 1441.75 | 1446.57 | 1450.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 1468.95 | 1449.93 | 1449.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 09:15:00 | 1468.95 | 1449.93 | 1449.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 13:15:00 | 1480.00 | 1464.74 | 1457.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 1491.20 | 1491.86 | 1483.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 1491.20 | 1491.86 | 1483.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1485.00 | 1490.48 | 1483.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1460.70 | 1490.48 | 1483.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1457.25 | 1483.84 | 1481.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1457.25 | 1483.84 | 1481.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1435.45 | 1474.16 | 1477.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1421.20 | 1452.93 | 1465.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1441.30 | 1439.85 | 1455.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 10:00:00 | 1441.30 | 1439.85 | 1455.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1450.35 | 1443.81 | 1453.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 1450.05 | 1443.81 | 1453.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1330.95 | 1302.37 | 1330.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:45:00 | 1329.75 | 1302.37 | 1330.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 1344.00 | 1310.69 | 1331.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 1344.00 | 1310.69 | 1331.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 1358.00 | 1320.16 | 1334.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:00:00 | 1358.00 | 1320.16 | 1334.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 14:15:00 | 1450.05 | 1346.13 | 1344.86 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1372.00 | 1392.12 | 1394.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1365.05 | 1386.70 | 1391.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 1417.00 | 1391.61 | 1392.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 1417.00 | 1391.61 | 1392.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 1417.00 | 1391.61 | 1392.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 1417.00 | 1391.61 | 1392.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 15:15:00 | 1412.90 | 1395.86 | 1394.45 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 11:15:00 | 1389.95 | 1397.68 | 1398.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1385.35 | 1394.08 | 1396.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 1313.85 | 1308.58 | 1336.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 1313.85 | 1308.58 | 1336.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1331.20 | 1314.57 | 1330.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 1331.20 | 1314.57 | 1330.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1334.05 | 1318.47 | 1330.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1334.05 | 1318.47 | 1330.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1328.05 | 1320.38 | 1330.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-29 12:45:00 | 1319.00 | 1319.22 | 1329.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 1414.60 | 1339.31 | 1335.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 1414.60 | 1339.31 | 1335.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 1477.20 | 1366.89 | 1347.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 15:15:00 | 1413.35 | 1414.43 | 1383.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:15:00 | 1441.90 | 1414.43 | 1383.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1466.30 | 1466.65 | 1436.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1457.85 | 1466.65 | 1436.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1441.65 | 1461.65 | 1436.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1446.15 | 1461.65 | 1436.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1451.40 | 1459.60 | 1438.21 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 1369.55 | 1427.52 | 1429.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 10:15:00 | 1362.05 | 1380.58 | 1399.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 1368.95 | 1367.53 | 1384.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 09:15:00 | 1372.40 | 1367.53 | 1384.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1381.00 | 1370.23 | 1384.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 1389.20 | 1370.23 | 1384.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1371.95 | 1370.57 | 1383.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 15:00:00 | 1366.50 | 1372.13 | 1380.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 09:15:00 | 1362.50 | 1372.11 | 1379.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 10:00:00 | 1363.45 | 1370.38 | 1378.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1298.17 | 1327.15 | 1340.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1294.38 | 1327.15 | 1340.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 1295.28 | 1327.15 | 1340.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-12 09:15:00 | 1229.85 | 1284.79 | 1310.41 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 1179.45 | 1159.44 | 1156.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 1193.90 | 1171.05 | 1163.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1186.00 | 1195.57 | 1181.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1186.00 | 1195.57 | 1181.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 1184.15 | 1193.28 | 1181.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 11:30:00 | 1196.70 | 1193.47 | 1182.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1202.30 | 1193.47 | 1182.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1173.75 | 1189.34 | 1185.00 | SL hit (close<static) qty=1.00 sl=1181.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 14:15:00 | 1174.50 | 1183.04 | 1183.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 14:15:00 | 1156.90 | 1173.97 | 1178.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 1100.35 | 1064.13 | 1084.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 09:15:00 | 1100.35 | 1064.13 | 1084.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 09:15:00 | 1100.35 | 1064.13 | 1084.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 10:00:00 | 1100.35 | 1064.13 | 1084.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1079.80 | 1067.26 | 1083.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:30:00 | 1074.15 | 1067.41 | 1082.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 1090.00 | 1082.17 | 1081.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 1090.00 | 1082.17 | 1081.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1098.15 | 1086.40 | 1083.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 1117.60 | 1127.76 | 1116.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1117.60 | 1127.76 | 1116.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1117.60 | 1127.76 | 1116.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1117.60 | 1127.76 | 1116.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1114.00 | 1125.01 | 1116.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 1114.00 | 1125.01 | 1116.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 1111.00 | 1122.20 | 1115.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 1111.00 | 1122.20 | 1115.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 1108.75 | 1119.51 | 1114.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:45:00 | 1108.10 | 1119.51 | 1114.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 1089.40 | 1109.28 | 1111.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 1072.90 | 1102.00 | 1107.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 1094.00 | 1087.29 | 1095.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 1089.00 | 1087.29 | 1095.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1086.00 | 1087.03 | 1095.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 1082.65 | 1087.03 | 1095.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:45:00 | 1078.65 | 1084.67 | 1093.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 12:15:00 | 1028.52 | 1041.67 | 1054.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 13:15:00 | 1024.72 | 1038.93 | 1051.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1081.40 | 1040.72 | 1048.78 | SL hit (close>ema200) qty=0.50 sl=1040.72 alert=retest2 |

### Cycle 61 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 1060.50 | 1053.72 | 1053.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 1117.80 | 1067.94 | 1059.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 1121.95 | 1122.29 | 1105.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 1162.15 | 1122.29 | 1105.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1158.80 | 1129.59 | 1110.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 1177.25 | 1129.59 | 1110.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 13:15:00 | 1294.98 | 1239.40 | 1191.80 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 1190.05 | 1202.33 | 1203.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 1178.60 | 1195.61 | 1200.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 09:15:00 | 1186.50 | 1178.51 | 1187.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 09:15:00 | 1186.50 | 1178.51 | 1187.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1186.50 | 1178.51 | 1187.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 1171.55 | 1179.74 | 1186.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:00:00 | 1170.85 | 1177.96 | 1184.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 1171.20 | 1174.86 | 1182.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:00:00 | 1173.35 | 1175.03 | 1180.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 1176.25 | 1175.34 | 1179.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 1179.20 | 1175.34 | 1179.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 1177.50 | 1175.77 | 1179.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 13:45:00 | 1175.65 | 1175.77 | 1179.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 14:15:00 | 1183.90 | 1177.40 | 1179.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 15:00:00 | 1183.90 | 1177.40 | 1179.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 1186.50 | 1179.22 | 1180.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 1155.15 | 1179.22 | 1180.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1167.65 | 1173.86 | 1177.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1167.65 | 1173.86 | 1177.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 1175.90 | 1174.27 | 1177.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:00:00 | 1175.90 | 1174.27 | 1177.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1175.05 | 1174.43 | 1177.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 12:45:00 | 1176.80 | 1174.43 | 1177.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1176.85 | 1174.91 | 1177.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:45:00 | 1178.85 | 1174.91 | 1177.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1181.20 | 1176.17 | 1177.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:45:00 | 1182.50 | 1176.17 | 1177.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-02 15:15:00 | 1188.20 | 1178.58 | 1178.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 1188.20 | 1178.58 | 1178.53 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 09:15:00 | 1175.40 | 1177.94 | 1178.25 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1183.00 | 1178.95 | 1178.68 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1143.00 | 1173.81 | 1176.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 1139.40 | 1162.31 | 1170.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1093.00 | 1086.81 | 1114.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1121.90 | 1086.81 | 1114.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1105.70 | 1090.59 | 1114.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 1098.90 | 1093.99 | 1113.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 1102.10 | 1093.99 | 1113.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 13:30:00 | 1101.15 | 1099.98 | 1111.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1084.85 | 1101.87 | 1110.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1139.90 | 1101.17 | 1103.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 1140.00 | 1108.94 | 1106.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1140.00 | 1108.94 | 1106.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 13:15:00 | 1146.40 | 1125.13 | 1115.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 1226.50 | 1228.07 | 1197.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:30:00 | 1223.30 | 1229.28 | 1200.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1225.80 | 1243.47 | 1239.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1225.80 | 1243.47 | 1239.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1235.80 | 1241.93 | 1238.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1225.60 | 1241.93 | 1238.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 12:15:00 | 1237.20 | 1240.45 | 1238.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 13:00:00 | 1237.20 | 1240.45 | 1238.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 1245.00 | 1241.36 | 1239.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:15:00 | 1246.90 | 1241.36 | 1239.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 14:45:00 | 1249.30 | 1242.77 | 1240.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 09:30:00 | 1246.90 | 1243.95 | 1241.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 14:15:00 | 1234.50 | 1240.55 | 1240.52 | SL hit (close<static) qty=1.00 sl=1235.50 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 15:15:00 | 1234.00 | 1239.24 | 1239.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 1187.30 | 1228.85 | 1235.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1212.00 | 1209.48 | 1219.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 09:45:00 | 1209.10 | 1209.48 | 1219.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1202.50 | 1202.46 | 1210.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:45:00 | 1196.10 | 1200.86 | 1207.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 15:15:00 | 1194.80 | 1200.41 | 1206.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 1176.60 | 1172.43 | 1171.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 1176.60 | 1172.43 | 1171.90 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1162.60 | 1170.33 | 1171.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 1153.90 | 1167.05 | 1169.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 1155.10 | 1144.33 | 1152.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 1155.10 | 1144.33 | 1152.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1155.10 | 1144.33 | 1152.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 1138.00 | 1144.79 | 1150.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1081.10 | 1128.12 | 1140.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1104.00 | 1103.59 | 1120.40 | SL hit (close>ema200) qty=0.50 sl=1103.59 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1182.60 | 1139.94 | 1134.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 1186.80 | 1149.31 | 1139.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1282.40 | 1284.74 | 1270.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 13:30:00 | 1280.00 | 1284.74 | 1270.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1270.00 | 1282.12 | 1272.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 1268.80 | 1282.12 | 1272.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1289.10 | 1283.52 | 1274.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 11:15:00 | 1314.40 | 1283.52 | 1274.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 1263.00 | 1275.03 | 1273.51 | SL hit (close<static) qty=1.00 sl=1264.10 alert=retest2 |

### Cycle 72 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1257.60 | 1271.55 | 1272.34 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1296.90 | 1273.99 | 1271.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 1307.80 | 1283.33 | 1275.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1269.00 | 1316.12 | 1303.99 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 12:15:00 | 1264.50 | 1290.54 | 1293.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 14:15:00 | 1255.20 | 1279.26 | 1287.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1228.00 | 1210.17 | 1230.39 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1254.70 | 1236.07 | 1235.09 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 1228.20 | 1236.86 | 1237.05 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 1240.00 | 1237.48 | 1237.32 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 09:15:00 | 1228.60 | 1235.87 | 1236.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 14:15:00 | 1221.00 | 1230.12 | 1232.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1250.40 | 1229.29 | 1229.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1250.40 | 1229.29 | 1229.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 1237.00 | 1230.84 | 1230.22 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 1226.00 | 1230.03 | 1230.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 15:15:00 | 1218.00 | 1223.08 | 1226.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1225.00 | 1223.46 | 1225.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:45:00 | 1225.10 | 1223.46 | 1225.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1223.00 | 1223.37 | 1225.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 1216.10 | 1222.64 | 1224.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1215.20 | 1220.80 | 1223.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1216.60 | 1219.06 | 1222.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 1155.29 | 1165.06 | 1172.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:15:00 | 1155.77 | 1165.06 | 1172.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1154.44 | 1161.77 | 1170.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 10:15:00 | 1156.30 | 1155.38 | 1163.36 | SL hit (close>ema200) qty=0.50 sl=1155.38 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1165.70 | 1157.78 | 1156.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1190.90 | 1165.63 | 1160.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1186.60 | 1188.61 | 1183.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1186.60 | 1188.61 | 1183.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1185.50 | 1187.98 | 1183.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 1185.30 | 1187.98 | 1183.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1182.30 | 1186.85 | 1183.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1182.30 | 1186.85 | 1183.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1171.00 | 1183.68 | 1182.33 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1172.40 | 1181.42 | 1181.42 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 1215.00 | 1182.34 | 1180.46 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1201.60 | 1207.78 | 1208.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 13:15:00 | 1199.00 | 1206.03 | 1207.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1205.20 | 1199.58 | 1203.18 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 14:15:00 | 1210.10 | 1202.93 | 1202.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 15:15:00 | 1223.00 | 1206.94 | 1204.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1206.50 | 1207.92 | 1205.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 1206.50 | 1207.92 | 1205.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1207.60 | 1207.85 | 1205.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:15:00 | 1206.10 | 1207.85 | 1205.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 1205.70 | 1207.42 | 1205.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 1206.60 | 1207.42 | 1205.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 1204.80 | 1206.90 | 1205.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 09:15:00 | 1209.70 | 1206.72 | 1205.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 1217.10 | 1224.15 | 1224.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 1217.10 | 1224.15 | 1224.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 14:15:00 | 1212.10 | 1219.10 | 1221.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1228.50 | 1220.21 | 1221.68 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1226.00 | 1223.15 | 1222.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1229.40 | 1224.26 | 1223.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1224.60 | 1225.27 | 1224.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1224.60 | 1225.27 | 1224.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1260.80 | 1232.37 | 1227.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1266.50 | 1243.19 | 1234.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 1279.40 | 1266.38 | 1258.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 14:45:00 | 1268.00 | 1268.94 | 1264.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1267.00 | 1268.94 | 1264.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1267.00 | 1268.55 | 1264.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:15:00 | 1283.30 | 1268.55 | 1264.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:45:00 | 1272.40 | 1269.93 | 1266.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 1271.50 | 1270.24 | 1266.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1273.00 | 1270.49 | 1267.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-24 09:15:00 | 1393.15 | 1315.68 | 1294.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 09:15:00 | 1414.90 | 1429.47 | 1429.73 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 1435.50 | 1426.56 | 1426.49 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 1408.30 | 1424.34 | 1425.69 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 1452.70 | 1427.71 | 1426.85 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1424.60 | 1439.58 | 1441.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 12:15:00 | 1419.90 | 1432.83 | 1437.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1446.60 | 1426.64 | 1432.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 1446.60 | 1426.64 | 1432.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1457.00 | 1432.71 | 1434.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:30:00 | 1447.90 | 1432.71 | 1434.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 1445.60 | 1435.29 | 1435.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:15:00 | 1441.00 | 1435.29 | 1435.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 13:15:00 | 1446.30 | 1435.64 | 1435.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 1446.30 | 1435.64 | 1435.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 1449.20 | 1438.35 | 1436.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1426.30 | 1437.16 | 1436.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 1426.30 | 1437.16 | 1436.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 1423.90 | 1434.51 | 1435.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 1416.60 | 1427.24 | 1431.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1454.00 | 1424.63 | 1428.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1454.00 | 1424.63 | 1428.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1441.30 | 1427.97 | 1429.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1436.00 | 1427.97 | 1429.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 1435.40 | 1429.45 | 1429.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 1433.40 | 1430.24 | 1430.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 13:15:00 | 1433.40 | 1430.24 | 1430.16 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 14:15:00 | 1415.60 | 1427.31 | 1428.84 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1447.60 | 1430.04 | 1429.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 1478.80 | 1447.29 | 1439.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 1467.50 | 1469.27 | 1456.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 10:00:00 | 1467.50 | 1469.27 | 1456.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1462.90 | 1468.00 | 1456.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 14:45:00 | 1468.90 | 1464.59 | 1458.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 10:30:00 | 1472.00 | 1465.89 | 1460.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 12:15:00 | 1452.70 | 1461.96 | 1459.56 | SL hit (close<static) qty=1.00 sl=1454.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1539.30 | 1574.86 | 1576.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 1536.10 | 1562.48 | 1570.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1541.50 | 1534.55 | 1551.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1541.50 | 1534.55 | 1551.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1555.70 | 1538.78 | 1551.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 1555.70 | 1538.78 | 1551.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1541.30 | 1539.28 | 1550.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:15:00 | 1531.30 | 1539.28 | 1550.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 15:15:00 | 1540.00 | 1539.05 | 1548.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 14:15:00 | 1583.00 | 1556.28 | 1553.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 1583.00 | 1556.28 | 1553.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1605.20 | 1581.80 | 1571.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 14:15:00 | 1579.20 | 1586.18 | 1578.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 15:00:00 | 1579.20 | 1586.18 | 1578.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1580.90 | 1585.12 | 1578.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1585.10 | 1585.12 | 1578.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1594.00 | 1584.60 | 1578.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 1575.40 | 1582.76 | 1578.59 | SL hit (close<static) qty=1.00 sl=1576.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 1549.70 | 1571.19 | 1573.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 1538.80 | 1564.71 | 1570.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1540.00 | 1537.78 | 1549.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 1541.50 | 1537.78 | 1549.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 1537.90 | 1519.49 | 1530.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:45:00 | 1547.10 | 1519.49 | 1530.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1568.10 | 1529.21 | 1534.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 1568.10 | 1529.21 | 1534.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 14:15:00 | 1572.00 | 1543.96 | 1540.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 1574.20 | 1555.48 | 1547.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 1613.50 | 1617.12 | 1593.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 1613.50 | 1617.12 | 1593.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1603.40 | 1614.38 | 1594.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1604.80 | 1614.38 | 1594.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1634.80 | 1643.75 | 1632.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1632.00 | 1643.75 | 1632.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1637.10 | 1642.42 | 1632.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1639.30 | 1634.90 | 1632.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:15:00 | 1640.90 | 1634.26 | 1632.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:45:00 | 1647.80 | 1637.41 | 1633.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1642.30 | 1649.97 | 1650.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 1642.30 | 1649.97 | 1650.09 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1663.70 | 1651.76 | 1650.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 12:15:00 | 1681.20 | 1664.69 | 1659.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1656.20 | 1665.39 | 1661.94 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 1638.50 | 1656.53 | 1658.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1626.90 | 1645.11 | 1651.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1582.20 | 1578.15 | 1600.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 1585.20 | 1578.15 | 1600.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1574.20 | 1573.63 | 1589.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1587.70 | 1573.63 | 1589.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1558.50 | 1554.24 | 1567.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 1564.90 | 1554.24 | 1567.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1555.20 | 1554.44 | 1566.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 1563.00 | 1554.44 | 1566.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1559.80 | 1555.64 | 1562.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1556.00 | 1555.64 | 1562.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1550.30 | 1554.57 | 1561.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1546.10 | 1552.90 | 1559.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1564.80 | 1550.10 | 1554.65 | SL hit (close>static) qty=1.00 sl=1563.80 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1561.90 | 1557.32 | 1556.80 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 1552.10 | 1556.28 | 1556.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1546.00 | 1554.22 | 1555.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1549.00 | 1542.60 | 1547.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 1553.00 | 1542.60 | 1547.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1547.70 | 1543.62 | 1547.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 1547.70 | 1543.62 | 1547.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1534.20 | 1541.74 | 1546.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 1545.00 | 1541.74 | 1546.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1536.90 | 1532.63 | 1539.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1536.90 | 1532.63 | 1539.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1532.30 | 1532.56 | 1538.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 1528.00 | 1532.56 | 1538.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1528.80 | 1533.76 | 1536.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:15:00 | 1528.60 | 1533.76 | 1536.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 1452.36 | 1476.47 | 1497.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:15:00 | 1452.17 | 1476.47 | 1497.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 11:15:00 | 1451.60 | 1469.57 | 1492.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1461.20 | 1453.23 | 1473.56 | SL hit (close>ema200) qty=0.50 sl=1453.23 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 1432.80 | 1426.87 | 1426.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 11:15:00 | 1449.00 | 1431.30 | 1428.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 10:15:00 | 1510.70 | 1512.96 | 1489.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:30:00 | 1509.10 | 1512.96 | 1489.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1526.20 | 1533.23 | 1524.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:00:00 | 1526.20 | 1533.23 | 1524.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1513.30 | 1529.24 | 1523.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 1513.30 | 1529.24 | 1523.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1516.30 | 1526.65 | 1522.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1523.00 | 1526.65 | 1522.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:00:00 | 1527.90 | 1526.90 | 1523.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1513.10 | 1523.05 | 1523.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 1513.10 | 1523.05 | 1523.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1502.10 | 1518.86 | 1521.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 1519.90 | 1518.31 | 1520.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:30:00 | 1521.00 | 1518.31 | 1520.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 1512.80 | 1517.21 | 1519.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 1518.40 | 1517.21 | 1519.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 1524.40 | 1518.65 | 1520.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 1524.40 | 1518.65 | 1520.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1521.00 | 1519.12 | 1520.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 1498.40 | 1519.12 | 1520.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:45:00 | 1510.00 | 1508.47 | 1512.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 12:15:00 | 1434.50 | 1462.30 | 1482.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-11 14:15:00 | 1423.48 | 1447.60 | 1471.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-13 09:15:00 | 1455.30 | 1435.44 | 1447.92 | SL hit (close>ema200) qty=0.50 sl=1435.44 alert=retest2 |

### Cycle 109 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 1209.90 | 1196.17 | 1196.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 10:15:00 | 1221.60 | 1209.99 | 1204.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1206.90 | 1215.07 | 1209.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1206.90 | 1215.07 | 1209.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1210.70 | 1214.20 | 1210.02 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1196.70 | 1206.09 | 1207.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1186.80 | 1200.78 | 1204.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1147.60 | 1145.68 | 1162.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:30:00 | 1153.50 | 1145.68 | 1162.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1157.30 | 1148.00 | 1161.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:45:00 | 1157.10 | 1148.00 | 1161.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1157.80 | 1149.21 | 1158.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1157.80 | 1149.21 | 1158.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1175.60 | 1154.49 | 1160.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 1175.60 | 1154.49 | 1160.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1176.70 | 1158.93 | 1161.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1206.40 | 1158.93 | 1161.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1193.00 | 1165.74 | 1164.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 1261.00 | 1211.67 | 1195.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 1235.20 | 1237.33 | 1219.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1235.20 | 1237.33 | 1219.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1217.80 | 1232.52 | 1224.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:15:00 | 1211.10 | 1232.52 | 1224.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1209.00 | 1227.82 | 1223.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1212.60 | 1227.82 | 1223.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1202.00 | 1219.45 | 1220.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1197.90 | 1210.06 | 1215.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1210.20 | 1210.09 | 1214.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 1213.60 | 1210.09 | 1214.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1209.20 | 1209.91 | 1214.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1198.50 | 1207.63 | 1212.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:45:00 | 1200.00 | 1205.90 | 1211.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1199.40 | 1207.17 | 1210.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 1198.00 | 1206.17 | 1209.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1204.20 | 1202.87 | 1206.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1262.10 | 1214.12 | 1209.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1262.10 | 1214.12 | 1209.57 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1220.70 | 1230.74 | 1231.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1211.70 | 1226.93 | 1229.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1215.60 | 1212.50 | 1218.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:45:00 | 1220.00 | 1212.50 | 1218.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1207.90 | 1211.37 | 1216.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:45:00 | 1200.00 | 1208.62 | 1215.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1140.00 | 1162.41 | 1180.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 15:15:00 | 1144.50 | 1142.70 | 1160.66 | SL hit (close>ema200) qty=0.50 sl=1142.70 alert=retest2 |

### Cycle 115 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 1055.30 | 1017.94 | 1013.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 1082.00 | 1030.75 | 1019.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 10:15:00 | 1052.60 | 1055.84 | 1038.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:45:00 | 1051.20 | 1055.84 | 1038.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1083.00 | 1063.50 | 1049.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 1095.70 | 1077.80 | 1064.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 14:30:00 | 1092.60 | 1085.34 | 1075.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 12:15:00 | 1064.80 | 1071.24 | 1071.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 1064.80 | 1071.24 | 1071.63 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1079.30 | 1072.20 | 1071.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 1086.00 | 1074.96 | 1073.24 | Break + close above crossover candle high |

### Cycle 118 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 1060.50 | 1072.07 | 1072.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 10:15:00 | 1049.10 | 1058.69 | 1063.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1044.00 | 1030.78 | 1040.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1044.00 | 1030.78 | 1040.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1046.20 | 1033.87 | 1040.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:30:00 | 1046.50 | 1033.87 | 1040.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 1057.30 | 1045.91 | 1044.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1068.50 | 1051.88 | 1047.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 1061.00 | 1061.16 | 1055.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 1051.10 | 1061.16 | 1055.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1050.00 | 1058.93 | 1054.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1047.40 | 1058.93 | 1054.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1048.80 | 1056.90 | 1054.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1048.80 | 1056.90 | 1054.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1050.80 | 1053.12 | 1052.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 1049.70 | 1053.12 | 1052.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1051.70 | 1052.84 | 1052.84 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 1053.40 | 1052.95 | 1052.89 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1037.30 | 1049.82 | 1051.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 1014.70 | 1032.51 | 1040.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1027.90 | 1026.39 | 1032.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 1027.90 | 1026.39 | 1032.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 10:15:00 | 1040.00 | 1029.11 | 1033.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 11:00:00 | 1040.00 | 1029.11 | 1033.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 11:15:00 | 1026.50 | 1028.59 | 1032.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 13:30:00 | 1025.50 | 1027.28 | 1031.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 11:00:00 | 1024.40 | 1027.12 | 1030.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1025.00 | 1026.54 | 1028.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 1025.40 | 1026.54 | 1028.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1025.40 | 1026.31 | 1028.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1038.50 | 1026.31 | 1028.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1032.30 | 1027.51 | 1028.85 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 1030.40 | 1029.39 | 1029.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1030.40 | 1029.39 | 1029.31 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1017.30 | 1026.98 | 1028.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1013.00 | 1024.18 | 1026.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1018.40 | 1014.89 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 1019.00 | 1014.89 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1012.00 | 1014.32 | 1017.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1018.00 | 1014.32 | 1017.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 999.00 | 1011.25 | 1016.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 994.60 | 1008.14 | 1014.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 1043.50 | 1011.72 | 1014.10 | SL hit (close>static) qty=1.00 sl=1020.60 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1041.00 | 1017.58 | 1016.54 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 12:15:00 | 1010.00 | 1018.15 | 1018.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 13:15:00 | 1008.70 | 1016.26 | 1017.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1023.40 | 1016.19 | 1017.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:45:00 | 1010.80 | 1015.26 | 1016.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 1008.50 | 1014.00 | 1015.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 960.26 | 997.89 | 1005.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 958.07 | 997.89 | 1005.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-04 11:15:00 | 909.72 | 941.00 | 966.79 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 930.00 | 914.12 | 912.46 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 900.75 | 912.35 | 912.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 880.60 | 903.79 | 908.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 909.95 | 903.82 | 907.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 909.95 | 903.82 | 907.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 909.00 | 904.86 | 907.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 902.30 | 905.41 | 907.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 15:00:00 | 896.05 | 886.61 | 890.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:30:00 | 903.30 | 892.71 | 893.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 905.00 | 895.17 | 894.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 10:15:00 | 905.00 | 895.17 | 894.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 974.85 | 915.82 | 904.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 14:15:00 | 997.40 | 1003.84 | 976.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 14:45:00 | 996.50 | 1003.84 | 976.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 1032.10 | 1056.72 | 1033.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:45:00 | 1032.50 | 1056.72 | 1033.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 1023.85 | 1050.14 | 1032.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 1023.85 | 1050.14 | 1032.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 1030.50 | 1046.21 | 1032.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:30:00 | 1026.60 | 1046.21 | 1032.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1023.00 | 1038.65 | 1031.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1097.05 | 1038.65 | 1031.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 10:15:00 | 1028.35 | 1055.46 | 1057.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1028.35 | 1055.46 | 1057.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 15:15:00 | 1019.45 | 1036.76 | 1046.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1033.00 | 996.14 | 1013.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1044.65 | 996.14 | 1013.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1029.30 | 1002.77 | 1015.01 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 1025.75 | 1021.90 | 1021.64 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 993.70 | 1017.19 | 1019.61 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1025.80 | 1017.96 | 1017.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1029.00 | 1020.16 | 1018.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 1026.75 | 1027.92 | 1024.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:15:00 | 1026.00 | 1027.92 | 1024.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1026.00 | 1027.54 | 1024.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1061.00 | 1027.54 | 1024.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-10 10:15:00 | 1167.10 | 1135.64 | 1102.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1201.20 | 1210.64 | 1211.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 1195.65 | 1207.64 | 1209.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1216.25 | 1206.70 | 1208.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:00:00 | 1216.25 | 1206.70 | 1208.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 1230.00 | 1211.36 | 1210.62 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 1202.50 | 1220.06 | 1221.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 1200.00 | 1213.80 | 1218.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1217.90 | 1209.82 | 1214.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1222.95 | 1209.82 | 1214.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1223.40 | 1212.54 | 1215.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1225.80 | 1212.54 | 1215.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1227.00 | 1218.50 | 1217.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1245.10 | 1225.98 | 1221.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1264.40 | 1274.37 | 1260.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1264.40 | 1274.37 | 1260.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1256.20 | 1270.73 | 1260.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 1256.20 | 1270.73 | 1260.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 1264.00 | 1269.39 | 1260.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 1259.05 | 1269.39 | 1260.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1238.00 | 1263.11 | 1258.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:15:00 | 1236.05 | 1263.11 | 1258.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1232.15 | 1256.92 | 1256.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:45:00 | 1229.25 | 1256.92 | 1256.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 1241.30 | 1253.79 | 1254.73 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 1260.60 | 1250.03 | 1248.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1269.50 | 1257.97 | 1253.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:45:00 | 1745.00 | 2024-05-28 13:15:00 | 1794.00 | STOP_HIT | 1.00 | 2.81% |
| SELL | retest2 | 2024-05-30 10:00:00 | 1781.00 | 2024-06-03 11:15:00 | 1799.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-05-30 11:00:00 | 1782.20 | 2024-06-03 11:15:00 | 1799.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-06-24 14:15:00 | 1744.65 | 2024-06-25 09:15:00 | 1818.10 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-07-03 11:15:00 | 1818.55 | 2024-07-04 11:15:00 | 1805.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-07-04 09:15:00 | 1822.35 | 2024-07-04 11:15:00 | 1805.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-07-16 11:45:00 | 1815.60 | 2024-07-19 09:15:00 | 1724.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 11:45:00 | 1815.60 | 2024-07-22 09:15:00 | 1767.50 | STOP_HIT | 0.50 | 2.65% |
| BUY | retest2 | 2024-09-04 12:30:00 | 1565.00 | 2024-09-04 14:15:00 | 1560.55 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-09-04 13:15:00 | 1568.00 | 2024-09-04 14:15:00 | 1560.55 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-09-19 15:15:00 | 1668.05 | 2024-09-25 12:15:00 | 1672.30 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2024-09-20 09:30:00 | 1668.35 | 2024-09-25 12:15:00 | 1672.30 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-09-20 11:45:00 | 1676.15 | 2024-09-25 12:15:00 | 1672.30 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-09-20 14:15:00 | 1666.00 | 2024-09-25 12:15:00 | 1672.30 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2024-09-24 09:15:00 | 1723.00 | 2024-09-25 12:15:00 | 1672.30 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2024-09-27 11:45:00 | 1654.00 | 2024-10-01 09:15:00 | 1688.15 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-09-30 10:30:00 | 1656.90 | 2024-10-01 09:15:00 | 1688.15 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-10-14 13:15:00 | 1751.10 | 2024-10-16 14:15:00 | 1710.50 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-10-14 14:00:00 | 1747.95 | 2024-10-16 14:15:00 | 1710.50 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-10-15 09:15:00 | 1750.65 | 2024-10-16 14:15:00 | 1710.50 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-10-25 09:30:00 | 1640.00 | 2024-10-30 10:15:00 | 1646.10 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-10-30 09:30:00 | 1640.85 | 2024-10-30 10:15:00 | 1646.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2024-11-19 12:30:00 | 1439.95 | 2024-11-25 11:15:00 | 1436.80 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2024-12-03 09:15:00 | 1577.65 | 2024-12-05 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-12-05 10:00:00 | 1572.60 | 2024-12-05 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-12-05 11:15:00 | 1573.50 | 2024-12-05 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1432.15 | 2025-01-01 09:15:00 | 1468.95 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-12-31 10:00:00 | 1441.75 | 2025-01-01 09:15:00 | 1468.95 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-01-29 12:45:00 | 1319.00 | 2025-01-30 09:15:00 | 1414.60 | STOP_HIT | 1.00 | -7.25% |
| SELL | retest2 | 2025-02-05 15:00:00 | 1366.50 | 2025-02-11 09:15:00 | 1298.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1362.50 | 2025-02-11 09:15:00 | 1294.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 10:00:00 | 1363.45 | 2025-02-11 09:15:00 | 1295.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 15:00:00 | 1366.50 | 2025-02-12 09:15:00 | 1229.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 09:15:00 | 1362.50 | 2025-02-12 09:15:00 | 1226.25 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 10:00:00 | 1363.45 | 2025-02-12 09:15:00 | 1227.11 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-21 11:30:00 | 1196.70 | 2025-02-24 09:15:00 | 1173.75 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-02-21 12:15:00 | 1202.30 | 2025-02-24 09:15:00 | 1173.75 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-03-04 11:30:00 | 1074.15 | 2025-03-05 14:15:00 | 1090.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1082.65 | 2025-03-17 12:15:00 | 1028.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 10:45:00 | 1078.65 | 2025-03-17 13:15:00 | 1024.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1082.65 | 2025-03-18 09:15:00 | 1081.40 | STOP_HIT | 0.50 | 0.12% |
| SELL | retest2 | 2025-03-12 10:45:00 | 1078.65 | 2025-03-18 09:15:00 | 1081.40 | STOP_HIT | 0.50 | -0.25% |
| SELL | retest2 | 2025-03-18 10:30:00 | 1075.65 | 2025-03-18 14:15:00 | 1060.50 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-03-21 10:15:00 | 1177.25 | 2025-03-24 13:15:00 | 1294.98 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-28 12:15:00 | 1171.55 | 2025-04-02 15:15:00 | 1188.20 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-03-28 13:00:00 | 1170.85 | 2025-04-02 15:15:00 | 1188.20 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-03-28 15:00:00 | 1171.20 | 2025-04-02 15:15:00 | 1188.20 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-04-01 11:00:00 | 1173.35 | 2025-04-02 15:15:00 | 1188.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-04-08 10:30:00 | 1098.90 | 2025-04-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2025-04-08 11:15:00 | 1102.10 | 2025-04-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-04-08 13:30:00 | 1101.15 | 2025-04-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -3.53% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1084.85 | 2025-04-11 10:15:00 | 1140.00 | STOP_HIT | 1.00 | -5.08% |
| BUY | retest2 | 2025-04-23 14:15:00 | 1246.90 | 2025-04-24 14:15:00 | 1234.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-04-23 14:45:00 | 1249.30 | 2025-04-24 14:15:00 | 1234.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-04-24 09:30:00 | 1246.90 | 2025-04-24 14:15:00 | 1234.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-04-29 12:45:00 | 1196.10 | 2025-05-06 11:15:00 | 1176.60 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-04-29 15:15:00 | 1194.80 | 2025-05-06 11:15:00 | 1176.60 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1138.00 | 2025-05-09 09:15:00 | 1081.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1138.00 | 2025-05-09 15:15:00 | 1104.00 | STOP_HIT | 0.50 | 2.99% |
| BUY | retest2 | 2025-05-20 11:15:00 | 1314.40 | 2025-05-20 15:15:00 | 1263.00 | STOP_HIT | 1.00 | -3.91% |
| SELL | retest2 | 2025-06-11 13:15:00 | 1216.10 | 2025-06-19 11:15:00 | 1155.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1215.20 | 2025-06-19 11:15:00 | 1155.77 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1216.60 | 2025-06-19 12:15:00 | 1154.44 | PARTIAL | 0.50 | 5.11% |
| SELL | retest2 | 2025-06-11 13:15:00 | 1216.10 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-06-11 15:15:00 | 1215.20 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1216.60 | 2025-06-20 10:15:00 | 1156.30 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1209.70 | 2025-07-14 11:15:00 | 1217.10 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1266.50 | 2025-07-24 09:15:00 | 1393.15 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-18 14:30:00 | 1279.40 | 2025-07-24 09:15:00 | 1393.70 | TARGET_HIT | 1.00 | 8.93% |
| BUY | retest2 | 2025-07-21 14:45:00 | 1268.00 | 2025-07-24 12:15:00 | 1407.34 | TARGET_HIT | 1.00 | 10.99% |
| BUY | retest2 | 2025-07-21 15:15:00 | 1267.00 | 2025-07-24 12:15:00 | 1394.80 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2025-07-22 09:15:00 | 1283.30 | 2025-07-24 12:15:00 | 1411.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 10:45:00 | 1272.40 | 2025-07-24 12:15:00 | 1399.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 12:00:00 | 1271.50 | 2025-07-24 12:15:00 | 1398.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-22 12:45:00 | 1273.00 | 2025-07-24 12:15:00 | 1400.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-07 12:15:00 | 1441.00 | 2025-08-07 13:15:00 | 1446.30 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1436.00 | 2025-08-11 13:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-11 13:00:00 | 1435.40 | 2025-08-11 13:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-08-14 14:45:00 | 1468.90 | 2025-08-18 12:15:00 | 1452.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-08-18 10:30:00 | 1472.00 | 2025-08-18 12:15:00 | 1452.70 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-19 09:30:00 | 1476.80 | 2025-08-25 11:15:00 | 1624.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-29 13:15:00 | 1531.30 | 2025-09-01 14:15:00 | 1583.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-08-29 15:15:00 | 1540.00 | 2025-09-01 14:15:00 | 1583.00 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1585.10 | 2025-09-04 10:15:00 | 1575.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1594.00 | 2025-09-04 10:15:00 | 1575.40 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1639.30 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-09-17 10:15:00 | 1640.90 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2025-09-17 10:45:00 | 1647.80 | 2025-09-19 14:15:00 | 1642.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1546.10 | 2025-10-06 09:15:00 | 1564.80 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-10-09 13:15:00 | 1528.00 | 2025-10-14 10:15:00 | 1452.36 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1528.80 | 2025-10-14 10:15:00 | 1452.17 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1528.60 | 2025-10-14 11:15:00 | 1451.60 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2025-10-09 13:15:00 | 1528.00 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.37% |
| SELL | retest2 | 2025-10-10 11:30:00 | 1528.80 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.42% |
| SELL | retest2 | 2025-10-10 12:15:00 | 1528.60 | 2025-10-15 09:15:00 | 1461.20 | STOP_HIT | 0.50 | 4.41% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1523.00 | 2025-11-04 15:15:00 | 1513.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-11-04 10:00:00 | 1527.90 | 2025-11-04 15:15:00 | 1513.10 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1498.40 | 2025-11-11 12:15:00 | 1434.50 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1510.00 | 2025-11-11 14:15:00 | 1423.48 | PARTIAL | 0.50 | 5.73% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1498.40 | 2025-11-13 09:15:00 | 1455.30 | STOP_HIT | 0.50 | 2.88% |
| SELL | retest2 | 2025-11-10 09:45:00 | 1510.00 | 2025-11-13 09:15:00 | 1455.30 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1198.50 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-12-30 12:45:00 | 1200.00 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.17% |
| SELL | retest2 | 2025-12-31 09:45:00 | 1199.40 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2025-12-31 12:15:00 | 1198.00 | 2026-01-02 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2026-01-08 10:45:00 | 1200.00 | 2026-01-12 09:15:00 | 1140.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:45:00 | 1200.00 | 2026-01-12 15:15:00 | 1144.50 | STOP_HIT | 0.50 | 4.63% |
| BUY | retest2 | 2026-02-01 09:15:00 | 1095.70 | 2026-02-02 12:15:00 | 1064.80 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-02-01 14:30:00 | 1092.60 | 2026-02-02 12:15:00 | 1064.80 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2026-02-16 13:30:00 | 1025.50 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-02-17 11:00:00 | 1024.40 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1025.00 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2026-02-17 15:15:00 | 1025.40 | 2026-02-19 09:15:00 | 1030.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-02-23 10:30:00 | 994.60 | 2026-02-23 13:15:00 | 1043.50 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1010.80 | 2026-03-02 09:15:00 | 960.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 1008.50 | 2026-03-02 09:15:00 | 958.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:45:00 | 1010.80 | 2026-03-04 11:15:00 | 909.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 1008.50 | 2026-03-04 13:15:00 | 907.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:00:00 | 902.30 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-03-16 15:00:00 | 896.05 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-17 09:30:00 | 903.30 | 2026-03-17 10:15:00 | 905.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1097.05 | 2026-03-27 10:15:00 | 1028.35 | STOP_HIT | 1.00 | -6.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1061.00 | 2026-04-10 10:15:00 | 1167.10 | TARGET_HIT | 1.00 | 10.00% |
