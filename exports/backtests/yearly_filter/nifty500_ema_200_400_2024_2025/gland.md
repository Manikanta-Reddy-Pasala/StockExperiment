# Gland Pharma Ltd. (GLAND)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1906.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 4 |
| TARGET_HIT | 6 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 70 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 55
- **Target hits / Stop hits / Partials:** 6 / 60 / 4
- **Avg / median % per leg:** -1.38% / -2.16%
- **Sum % (uncompounded):** -96.85%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 7 | 23.3% | 3 | 27 | 0 | -1.11% | -33.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 7 | 23.3% | 3 | 27 | 0 | -1.11% | -33.3% |
| SELL (all) | 40 | 8 | 20.0% | 3 | 33 | 4 | -1.59% | -63.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 40 | 8 | 20.0% | 3 | 33 | 4 | -1.59% | -63.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 70 | 15 | 21.4% | 6 | 60 | 4 | -1.38% | -96.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 1852.00 | 1793.85 | 1793.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 1860.05 | 1807.10 | 1800.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 1800.00 | 1828.73 | 1814.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 1800.00 | 1828.73 | 1814.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 1800.00 | 1828.73 | 1814.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 1879.95 | 1821.94 | 1815.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 12:15:00 | 1879.65 | 1822.41 | 1815.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 09:30:00 | 1886.35 | 1825.52 | 1817.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-09 13:15:00 | 2067.95 | 1831.70 | 1820.61 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 09:15:00 | 1879.25 | 1922.75 | 1922.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 10:15:00 | 1866.10 | 1922.18 | 1922.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1785.50 | 1696.51 | 1763.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1785.50 | 1696.51 | 1763.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1785.50 | 1696.51 | 1763.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 1785.50 | 1696.51 | 1763.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1805.80 | 1697.60 | 1764.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 1805.80 | 1697.60 | 1764.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1807.00 | 1698.69 | 1764.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:30:00 | 1806.50 | 1698.69 | 1764.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 1761.40 | 1733.29 | 1771.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 09:15:00 | 1742.20 | 1733.29 | 1771.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 15:15:00 | 1777.00 | 1734.20 | 1769.53 | SL hit (close>static) qty=1.00 sl=1772.85 alert=retest2 |

### Cycle 3 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 1842.50 | 1776.03 | 1775.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1900.05 | 1779.73 | 1777.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 1799.60 | 1804.77 | 1791.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 1799.60 | 1804.77 | 1791.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 1799.60 | 1804.77 | 1791.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:45:00 | 1787.35 | 1804.77 | 1791.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 1804.35 | 1804.76 | 1791.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 1799.00 | 1804.76 | 1791.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 11:15:00 | 1784.75 | 1804.56 | 1791.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:00:00 | 1784.75 | 1804.56 | 1791.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 1786.20 | 1804.38 | 1791.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 12:30:00 | 1787.80 | 1804.38 | 1791.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 1664.90 | 1779.99 | 1780.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1639.10 | 1754.56 | 1766.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 1551.60 | 1545.01 | 1614.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-24 12:45:00 | 1545.40 | 1545.01 | 1614.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 1582.70 | 1544.86 | 1612.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 1606.55 | 1544.86 | 1612.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 1609.60 | 1545.50 | 1612.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 10:45:00 | 1612.95 | 1545.50 | 1612.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 1603.60 | 1546.08 | 1612.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 12:45:00 | 1582.50 | 1546.47 | 1612.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 15:15:00 | 1577.00 | 1547.49 | 1612.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 09:15:00 | 1645.00 | 1548.75 | 1612.54 | SL hit (close>static) qty=1.00 sl=1620.80 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 1624.30 | 1520.47 | 1520.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 1626.80 | 1522.51 | 1521.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 1932.40 | 1935.96 | 1834.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:45:00 | 1933.30 | 1935.96 | 1834.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 1900.80 | 1937.42 | 1874.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 1909.80 | 1919.72 | 1875.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1870.20 | 1918.46 | 1876.20 | SL hit (close<static) qty=1.00 sl=1873.60 alert=retest2 |

### Cycle 6 — SELL (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 12:15:00 | 1849.00 | 1918.94 | 1919.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 14:15:00 | 1838.10 | 1917.46 | 1918.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 12:15:00 | 1721.50 | 1718.61 | 1776.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 13:00:00 | 1721.50 | 1718.61 | 1776.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1734.80 | 1713.25 | 1763.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1721.80 | 1713.25 | 1763.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 1722.50 | 1712.27 | 1753.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 15:15:00 | 1713.90 | 1706.47 | 1744.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1811.90 | 1704.09 | 1739.17 | SL hit (close>static) qty=1.00 sl=1790.40 alert=retest2 |

### Cycle 7 — BUY (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 12:15:00 | 1836.40 | 1766.11 | 1766.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 1842.60 | 1766.87 | 1766.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 1773.60 | 1790.72 | 1779.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 1775.00 | 1790.72 | 1779.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1781.50 | 1790.63 | 1779.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 12:15:00 | 1800.40 | 1790.68 | 1779.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 12:00:00 | 1802.00 | 1790.60 | 1779.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:45:00 | 1802.10 | 1790.90 | 1780.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 10:00:00 | 1800.00 | 1802.66 | 1788.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1783.60 | 1808.92 | 1793.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 1769.10 | 1808.17 | 1793.45 | SL hit (close<static) qty=1.00 sl=1770.80 alert=retest2 |

### Cycle 8 — SELL (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 14:15:00 | 1681.40 | 1780.28 | 1780.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 1669.70 | 1767.50 | 1774.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 10:15:00 | 1718.00 | 1701.33 | 1733.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-25 10:45:00 | 1719.40 | 1701.33 | 1733.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 1707.00 | 1701.88 | 1732.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 1681.70 | 1701.88 | 1732.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:45:00 | 1679.30 | 1701.59 | 1731.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1673.30 | 1702.55 | 1730.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 1680.10 | 1701.76 | 1729.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1723.80 | 1701.63 | 1728.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 10:45:00 | 1728.40 | 1701.63 | 1728.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 1707.20 | 1701.69 | 1728.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 1703.50 | 1701.71 | 1728.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 12:45:00 | 1702.70 | 1701.50 | 1727.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 12:15:00 | 1737.80 | 1703.06 | 1727.25 | SL hit (close>static) qty=1.00 sl=1736.00 alert=retest2 |

### Cycle 9 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1793.60 | 1741.23 | 1741.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1813.20 | 1741.95 | 1741.49 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-28 14:45:00 | 1813.95 | 2024-05-29 10:15:00 | 1840.90 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-05-28 15:15:00 | 1811.00 | 2024-05-29 10:15:00 | 1840.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-07-08 10:30:00 | 1879.95 | 2024-07-09 13:15:00 | 2067.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-08 12:15:00 | 1879.65 | 2024-07-09 13:15:00 | 2067.62 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-09 09:30:00 | 1886.35 | 2024-07-09 13:15:00 | 2074.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 12:30:00 | 1879.00 | 2024-09-09 13:15:00 | 1899.05 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2024-09-09 10:30:00 | 1949.40 | 2024-09-09 13:15:00 | 1899.05 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-09-09 12:00:00 | 1943.25 | 2024-09-11 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2024-09-10 10:15:00 | 1946.05 | 2024-09-11 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-09-10 11:00:00 | 1944.50 | 2024-09-13 09:15:00 | 1879.25 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-11-13 09:15:00 | 1742.20 | 2024-11-14 15:15:00 | 1777.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-11-18 09:30:00 | 1730.85 | 2024-11-18 14:15:00 | 1775.20 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2024-11-21 09:15:00 | 1745.00 | 2024-11-21 10:15:00 | 1782.70 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2024-11-26 15:00:00 | 1739.85 | 2024-12-02 11:15:00 | 1780.85 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-12-10 12:30:00 | 1773.05 | 2024-12-12 09:15:00 | 1806.50 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-12-11 14:15:00 | 1772.40 | 2024-12-12 09:15:00 | 1806.50 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-12-11 14:45:00 | 1773.15 | 2024-12-12 09:15:00 | 1806.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-12-11 15:15:00 | 1770.55 | 2024-12-12 09:15:00 | 1806.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-12-12 12:30:00 | 1796.05 | 2025-01-01 11:15:00 | 1820.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-12-12 14:30:00 | 1795.00 | 2025-01-02 11:15:00 | 1842.50 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-12-26 10:15:00 | 1786.60 | 2025-01-02 11:15:00 | 1842.50 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2024-12-27 13:30:00 | 1794.80 | 2025-01-02 11:15:00 | 1842.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-01-01 09:15:00 | 1768.15 | 2025-01-02 11:15:00 | 1842.50 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2025-02-25 12:45:00 | 1582.50 | 2025-02-27 09:15:00 | 1645.00 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-02-25 15:15:00 | 1577.00 | 2025-02-27 09:15:00 | 1645.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2025-02-28 10:00:00 | 1566.40 | 2025-02-28 12:15:00 | 1506.56 | PARTIAL | 0.50 | 3.82% |
| SELL | retest2 | 2025-02-28 10:00:00 | 1566.40 | 2025-02-28 15:15:00 | 1556.00 | STOP_HIT | 0.50 | 0.66% |
| SELL | retest2 | 2025-02-28 11:45:00 | 1585.85 | 2025-03-06 10:15:00 | 1625.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-03-12 10:15:00 | 1593.60 | 2025-03-21 11:15:00 | 1623.45 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-03-21 10:45:00 | 1598.90 | 2025-03-21 11:15:00 | 1623.45 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-03-21 15:00:00 | 1579.65 | 2025-03-21 15:15:00 | 1684.00 | STOP_HIT | 1.00 | -6.61% |
| SELL | retest2 | 2025-03-28 12:30:00 | 1592.15 | 2025-04-04 09:15:00 | 1512.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1570.10 | 2025-04-04 09:15:00 | 1491.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 12:45:00 | 1577.70 | 2025-04-04 09:15:00 | 1498.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 12:30:00 | 1592.15 | 2025-04-07 09:15:00 | 1432.94 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1570.10 | 2025-04-07 09:15:00 | 1413.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-03 12:45:00 | 1577.70 | 2025-04-07 09:15:00 | 1419.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-22 09:45:00 | 1578.00 | 2025-06-03 09:15:00 | 1634.40 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2025-05-29 13:15:00 | 1581.00 | 2025-06-03 09:15:00 | 1634.40 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2025-09-05 09:30:00 | 1909.80 | 2025-09-08 09:15:00 | 1870.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-09 10:15:00 | 1909.00 | 2025-10-10 13:15:00 | 1924.70 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-09-09 13:00:00 | 1908.10 | 2025-10-10 13:15:00 | 1924.70 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-09-09 13:45:00 | 1905.30 | 2025-10-20 12:15:00 | 1924.10 | STOP_HIT | 1.00 | 0.99% |
| BUY | retest2 | 2025-09-26 13:00:00 | 1957.00 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-09-29 10:30:00 | 1942.50 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-09-29 15:00:00 | 2023.40 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1946.70 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-10-08 14:30:00 | 1959.80 | 2025-10-27 10:15:00 | 1920.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-09 09:45:00 | 1946.80 | 2025-11-04 09:15:00 | 1890.90 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-10-17 14:00:00 | 1943.50 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -4.07% |
| BUY | retest2 | 2025-10-23 09:45:00 | 1947.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-10-24 09:15:00 | 1934.90 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.64% |
| BUY | retest2 | 2025-10-24 10:30:00 | 1936.80 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-10-24 13:30:00 | 1933.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-10-24 14:00:00 | 1933.70 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-11-03 09:45:00 | 1994.00 | 2025-11-07 10:15:00 | 1864.40 | STOP_HIT | 1.00 | -6.50% |
| SELL | retest2 | 2026-01-08 10:15:00 | 1721.80 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.23% |
| SELL | retest2 | 2026-01-16 13:00:00 | 1722.50 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.19% |
| SELL | retest2 | 2026-01-22 15:15:00 | 1713.90 | 2026-01-29 10:15:00 | 1811.90 | STOP_HIT | 1.00 | -5.72% |
| BUY | retest2 | 2026-02-13 12:15:00 | 1800.40 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-16 12:00:00 | 1802.00 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-16 14:45:00 | 1802.10 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-02-24 10:00:00 | 1800.00 | 2026-03-02 11:15:00 | 1769.10 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-03-02 14:45:00 | 1792.70 | 2026-03-04 09:15:00 | 1745.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-03-30 09:15:00 | 1681.70 | 2026-04-08 12:15:00 | 1737.80 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2026-03-30 09:45:00 | 1679.30 | 2026-04-08 12:15:00 | 1737.80 | STOP_HIT | 1.00 | -3.48% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1673.30 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -8.17% |
| SELL | retest2 | 2026-04-02 13:00:00 | 1680.10 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -7.73% |
| SELL | retest2 | 2026-04-06 13:00:00 | 1703.50 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -6.25% |
| SELL | retest2 | 2026-04-07 12:45:00 | 1702.70 | 2026-04-17 12:15:00 | 1810.00 | STOP_HIT | 1.00 | -6.30% |
