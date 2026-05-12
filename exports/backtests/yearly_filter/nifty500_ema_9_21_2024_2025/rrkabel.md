# R R Kabel Ltd. (RRKABEL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-11 15:15:00 (3717 bars)
- **Last close:** 1928.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 128 |
| ALERT1 | 93 |
| ALERT2 | 93 |
| ALERT2_SKIP | 50 |
| ALERT3 | 256 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 167 |
| PARTIAL | 21 |
| TARGET_HIT | 11 |
| STOP_HIT | 159 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 191 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 90 / 101
- **Target hits / Stop hits / Partials:** 11 / 159 / 21
- **Avg / median % per leg:** 0.76% / -0.22%
- **Sum % (uncompounded):** 145.68%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 92 | 29 | 31.5% | 8 | 84 | 0 | -0.13% | -11.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.11% | -3.3% |
| BUY @ 3rd Alert (retest2) | 89 | 29 | 32.6% | 8 | 81 | 0 | -0.09% | -8.4% |
| SELL (all) | 99 | 61 | 61.6% | 3 | 75 | 21 | 1.59% | 157.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 99 | 61 | 61.6% | 3 | 75 | 21 | 1.59% | 157.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.11% | -3.3% |
| retest2 (combined) | 188 | 90 | 47.9% | 11 | 156 | 21 | 0.79% | 149.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 15:15:00 | 1742.00 | 1751.84 | 1751.93 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 09:15:00 | 1770.50 | 1755.57 | 1753.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 09:15:00 | 1810.90 | 1771.22 | 1763.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 15:15:00 | 1799.95 | 1812.81 | 1801.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 15:15:00 | 1799.95 | 1812.81 | 1801.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 1799.95 | 1812.81 | 1801.80 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 13:15:00 | 1740.00 | 1788.83 | 1793.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 1729.00 | 1776.86 | 1787.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 10:15:00 | 1736.90 | 1735.30 | 1753.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 11:00:00 | 1736.90 | 1735.30 | 1753.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1735.55 | 1735.35 | 1751.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:45:00 | 1738.70 | 1735.35 | 1751.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1753.60 | 1739.00 | 1751.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:00:00 | 1753.60 | 1739.00 | 1751.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 1770.55 | 1745.31 | 1753.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 1770.55 | 1745.31 | 1753.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 1712.40 | 1738.73 | 1749.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:30:00 | 1773.90 | 1738.73 | 1749.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1750.10 | 1737.21 | 1746.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:30:00 | 1708.40 | 1730.60 | 1742.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 1658.75 | 1726.39 | 1736.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1622.98 | 1690.81 | 1717.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1575.81 | 1690.81 | 1717.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 1537.56 | 1650.95 | 1697.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 1677.60 | 1658.24 | 1656.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 1687.15 | 1672.53 | 1664.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 1745.05 | 1745.56 | 1726.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 1745.05 | 1745.56 | 1726.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 1727.00 | 1740.54 | 1727.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1763.65 | 1740.54 | 1727.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 1742.65 | 1747.37 | 1738.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 09:15:00 | 1750.95 | 1757.44 | 1757.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 09:15:00 | 1750.95 | 1757.44 | 1757.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 09:15:00 | 1737.60 | 1750.35 | 1753.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 13:15:00 | 1749.45 | 1745.89 | 1749.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 13:15:00 | 1749.45 | 1745.89 | 1749.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1749.45 | 1745.89 | 1749.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:45:00 | 1751.65 | 1745.89 | 1749.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1753.45 | 1747.40 | 1750.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 14:45:00 | 1754.70 | 1747.40 | 1750.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 1753.00 | 1748.52 | 1750.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:15:00 | 1725.50 | 1748.52 | 1750.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 14:15:00 | 1756.70 | 1742.06 | 1745.03 | SL hit (close>static) qty=1.00 sl=1754.95 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1759.95 | 1746.75 | 1746.72 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 1746.90 | 1752.48 | 1752.78 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 1760.55 | 1753.19 | 1752.94 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 12:15:00 | 1741.65 | 1751.24 | 1752.42 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 1768.00 | 1752.68 | 1752.08 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 1741.00 | 1753.57 | 1753.81 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 14:15:00 | 1755.55 | 1753.76 | 1753.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1800.55 | 1763.59 | 1758.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 1792.20 | 1795.45 | 1781.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:30:00 | 1794.75 | 1795.45 | 1781.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 11:15:00 | 1783.20 | 1793.00 | 1782.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 11:45:00 | 1783.35 | 1793.00 | 1782.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1784.05 | 1791.21 | 1782.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 12:30:00 | 1783.10 | 1791.21 | 1782.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1784.95 | 1788.89 | 1782.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 1780.10 | 1788.89 | 1782.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1785.00 | 1788.11 | 1782.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 1816.45 | 1788.11 | 1782.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:30:00 | 1800.05 | 1791.86 | 1785.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-08 11:15:00 | 1816.10 | 1800.16 | 1793.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 11:00:00 | 1799.50 | 1817.05 | 1813.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 1820.00 | 1819.89 | 1816.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-11 12:15:00 | 1809.65 | 1814.38 | 1814.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 12:15:00 | 1809.65 | 1814.38 | 1814.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 1802.00 | 1808.83 | 1811.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 10:15:00 | 1782.90 | 1779.02 | 1785.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 11:00:00 | 1782.90 | 1779.02 | 1785.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1784.35 | 1779.32 | 1784.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 1785.50 | 1779.32 | 1784.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1775.25 | 1778.51 | 1783.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 1771.00 | 1776.94 | 1782.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 15:00:00 | 1770.65 | 1776.94 | 1782.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 1787.95 | 1778.71 | 1782.54 | SL hit (close>static) qty=1.00 sl=1784.95 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 1739.00 | 1737.22 | 1737.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 10:15:00 | 1751.25 | 1741.35 | 1739.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 12:15:00 | 1740.40 | 1743.40 | 1740.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 1740.40 | 1743.40 | 1740.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 1740.40 | 1743.40 | 1740.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 1740.40 | 1743.40 | 1740.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 1747.15 | 1744.15 | 1741.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1804.40 | 1751.68 | 1747.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:15:00 | 1779.10 | 1784.28 | 1777.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 1728.95 | 1780.33 | 1780.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 09:15:00 | 1728.95 | 1780.33 | 1780.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 1707.20 | 1741.48 | 1757.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 1665.40 | 1662.00 | 1684.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 13:00:00 | 1665.40 | 1662.00 | 1684.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1687.15 | 1661.82 | 1675.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 1682.65 | 1661.82 | 1675.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1687.80 | 1667.02 | 1676.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 1688.05 | 1667.02 | 1676.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1702.65 | 1681.37 | 1681.23 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 1656.00 | 1677.87 | 1680.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 14:15:00 | 1645.75 | 1664.58 | 1671.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1607.80 | 1586.52 | 1600.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1607.80 | 1586.52 | 1600.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1607.80 | 1586.52 | 1600.28 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 1638.10 | 1610.57 | 1608.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 1665.95 | 1628.72 | 1617.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 1634.70 | 1637.34 | 1627.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 15:00:00 | 1634.70 | 1637.34 | 1627.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1654.65 | 1663.28 | 1653.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 1654.65 | 1663.28 | 1653.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1640.55 | 1658.74 | 1652.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1640.55 | 1658.74 | 1652.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1645.05 | 1656.00 | 1651.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 15:15:00 | 1652.00 | 1656.00 | 1651.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-22 11:15:00 | 1639.45 | 1648.19 | 1649.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 11:15:00 | 1639.45 | 1648.19 | 1649.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 1633.95 | 1645.34 | 1647.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 1645.70 | 1640.46 | 1644.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1645.70 | 1640.46 | 1644.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1645.70 | 1640.46 | 1644.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1645.70 | 1640.46 | 1644.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1640.75 | 1640.52 | 1643.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1644.50 | 1640.52 | 1643.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1639.50 | 1638.85 | 1641.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1619.00 | 1638.85 | 1641.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 1616.00 | 1634.28 | 1639.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 13:45:00 | 1606.85 | 1614.58 | 1620.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:30:00 | 1607.20 | 1609.91 | 1610.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 15:15:00 | 1615.00 | 1611.61 | 1611.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 15:15:00 | 1615.00 | 1611.61 | 1611.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1624.55 | 1614.20 | 1612.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 1613.40 | 1615.21 | 1613.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 11:15:00 | 1613.40 | 1615.21 | 1613.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1613.40 | 1615.21 | 1613.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:45:00 | 1613.10 | 1615.21 | 1613.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1613.15 | 1614.80 | 1613.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1611.10 | 1614.80 | 1613.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1623.10 | 1616.46 | 1614.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:15:00 | 1624.45 | 1616.46 | 1614.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 1604.05 | 1614.95 | 1614.18 | SL hit (close<static) qty=1.00 sl=1610.25 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 1601.95 | 1612.35 | 1613.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 11:15:00 | 1600.35 | 1609.95 | 1611.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 1607.15 | 1606.90 | 1609.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 1607.15 | 1606.90 | 1609.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1607.15 | 1606.90 | 1609.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 14:45:00 | 1608.90 | 1606.90 | 1609.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1601.05 | 1605.26 | 1608.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 10:30:00 | 1596.80 | 1604.41 | 1607.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 11:15:00 | 1595.00 | 1604.41 | 1607.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 14:15:00 | 1588.30 | 1598.87 | 1604.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 1591.85 | 1597.46 | 1602.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1593.55 | 1595.33 | 1600.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 10:15:00 | 1585.00 | 1595.33 | 1600.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 12:00:00 | 1583.45 | 1591.53 | 1598.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 10:15:00 | 1585.00 | 1584.87 | 1591.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 15:15:00 | 1578.00 | 1566.89 | 1566.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 1578.00 | 1566.89 | 1566.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 12:15:00 | 1606.60 | 1580.59 | 1573.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 13:15:00 | 1659.60 | 1660.41 | 1629.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 13:30:00 | 1655.00 | 1660.41 | 1629.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1656.30 | 1656.49 | 1650.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:45:00 | 1660.10 | 1656.60 | 1650.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 12:45:00 | 1660.05 | 1660.42 | 1653.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:00:00 | 1658.85 | 1662.34 | 1656.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:15:00 | 1663.25 | 1661.78 | 1658.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1667.45 | 1662.91 | 1658.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:00:00 | 1671.00 | 1665.59 | 1662.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 10:30:00 | 1670.95 | 1669.91 | 1665.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 11:00:00 | 1672.20 | 1669.91 | 1665.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 11:15:00 | 1678.15 | 1692.78 | 1693.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 1678.15 | 1692.78 | 1693.63 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 1716.40 | 1695.50 | 1692.84 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 13:15:00 | 1679.60 | 1691.51 | 1692.50 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 1700.30 | 1693.27 | 1693.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 1706.35 | 1695.89 | 1694.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 14:15:00 | 1730.40 | 1735.72 | 1725.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 14:15:00 | 1730.40 | 1735.72 | 1725.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1730.40 | 1735.72 | 1725.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:30:00 | 1725.00 | 1735.72 | 1725.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1718.85 | 1732.39 | 1725.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 1726.75 | 1732.39 | 1725.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 15:15:00 | 1723.50 | 1726.70 | 1725.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:45:00 | 1723.25 | 1725.46 | 1724.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 10:30:00 | 1732.00 | 1728.41 | 1726.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 1738.00 | 1734.73 | 1730.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 1722.10 | 1734.73 | 1730.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 1717.45 | 1731.28 | 1729.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:30:00 | 1713.05 | 1731.28 | 1729.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 1706.90 | 1726.40 | 1727.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1706.90 | 1726.40 | 1727.45 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 1772.80 | 1729.54 | 1727.76 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 1730.70 | 1740.27 | 1741.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 1727.80 | 1737.78 | 1740.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 09:15:00 | 1746.25 | 1733.64 | 1736.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 09:15:00 | 1746.25 | 1733.64 | 1736.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 1746.25 | 1733.64 | 1736.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:00:00 | 1746.25 | 1733.64 | 1736.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 1735.00 | 1733.91 | 1736.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-11 11:45:00 | 1725.00 | 1731.84 | 1735.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 1740.70 | 1733.21 | 1732.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 12:15:00 | 1740.70 | 1733.21 | 1732.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 1751.85 | 1739.79 | 1736.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 1769.75 | 1772.58 | 1759.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:00:00 | 1769.75 | 1772.58 | 1759.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 1780.00 | 1772.85 | 1761.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:30:00 | 1764.00 | 1772.85 | 1761.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 1756.60 | 1769.60 | 1761.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 1756.60 | 1769.60 | 1761.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 13:15:00 | 1761.65 | 1768.01 | 1761.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 1786.00 | 1771.61 | 1763.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 13:15:00 | 1752.55 | 1762.48 | 1762.16 | SL hit (close<static) qty=1.00 sl=1755.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 1733.25 | 1756.64 | 1759.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1699.75 | 1741.80 | 1752.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 1726.10 | 1724.44 | 1735.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 1726.10 | 1724.44 | 1735.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1726.10 | 1724.44 | 1735.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:30:00 | 1706.50 | 1718.05 | 1730.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-23 09:15:00 | 1621.17 | 1657.38 | 1681.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-25 09:15:00 | 1535.85 | 1595.10 | 1621.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 32 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 1512.25 | 1487.21 | 1486.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1532.00 | 1507.20 | 1497.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1500.90 | 1510.16 | 1501.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1500.90 | 1510.16 | 1501.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1500.90 | 1510.16 | 1501.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 1502.60 | 1510.16 | 1501.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1523.45 | 1512.82 | 1503.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 1584.15 | 1516.75 | 1508.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1542.70 | 1523.22 | 1517.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-06 12:00:00 | 1525.15 | 1528.43 | 1522.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 11:15:00 | 1506.85 | 1531.99 | 1535.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2024-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 11:15:00 | 1506.85 | 1531.99 | 1535.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 12:15:00 | 1494.00 | 1524.39 | 1531.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 1528.15 | 1512.27 | 1522.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 1528.15 | 1512.27 | 1522.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 1528.15 | 1512.27 | 1522.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 1528.15 | 1512.27 | 1522.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1520.00 | 1513.82 | 1521.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:45:00 | 1507.15 | 1512.02 | 1520.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 15:15:00 | 1537.00 | 1503.72 | 1506.12 | SL hit (close>static) qty=1.00 sl=1527.70 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 09:15:00 | 1550.00 | 1512.98 | 1510.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 1559.50 | 1537.20 | 1527.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1545.60 | 1545.91 | 1536.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 14:30:00 | 1548.80 | 1545.91 | 1536.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 1530.20 | 1542.78 | 1536.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 1523.65 | 1542.78 | 1536.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 1538.20 | 1541.86 | 1536.63 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 1515.00 | 1530.56 | 1532.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 09:15:00 | 1501.00 | 1522.80 | 1528.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 1504.60 | 1503.19 | 1512.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-25 11:00:00 | 1504.60 | 1503.19 | 1512.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 1470.00 | 1468.14 | 1479.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 13:30:00 | 1465.30 | 1468.46 | 1478.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 14:15:00 | 1465.00 | 1468.46 | 1478.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 1464.70 | 1467.76 | 1474.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:00:00 | 1463.80 | 1466.97 | 1473.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1465.00 | 1446.33 | 1453.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 1465.00 | 1446.33 | 1453.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1458.75 | 1448.81 | 1453.83 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-02 13:15:00 | 1467.00 | 1456.79 | 1456.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 1467.00 | 1456.79 | 1456.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 10:15:00 | 1473.60 | 1461.36 | 1458.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 13:15:00 | 1462.50 | 1464.28 | 1461.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 14:00:00 | 1462.50 | 1464.28 | 1461.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1462.65 | 1463.95 | 1461.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:00:00 | 1462.65 | 1463.95 | 1461.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1463.50 | 1463.86 | 1461.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 1477.10 | 1463.86 | 1461.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1461.00 | 1463.29 | 1461.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 1460.05 | 1463.29 | 1461.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1459.25 | 1462.48 | 1461.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 1458.60 | 1462.48 | 1461.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1459.05 | 1461.79 | 1461.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1459.25 | 1461.79 | 1461.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 12:15:00 | 1455.05 | 1460.45 | 1460.49 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 13:15:00 | 1462.05 | 1460.77 | 1460.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 1497.65 | 1468.14 | 1464.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 15:15:00 | 1482.00 | 1485.24 | 1477.52 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:15:00 | 1499.00 | 1485.24 | 1477.52 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:15:00 | 1491.50 | 1486.15 | 1479.29 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1479.80 | 1485.13 | 1481.54 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-06 15:15:00 | 1479.80 | 1485.13 | 1481.54 | SL hit (close<ema400) qty=1.00 sl=1481.54 alert=retest1 |

### Cycle 39 — SELL (started 2024-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 10:15:00 | 1461.65 | 1477.01 | 1478.26 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 1500.00 | 1477.26 | 1475.49 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 1463.60 | 1476.23 | 1477.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 1458.00 | 1468.05 | 1472.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 14:15:00 | 1409.20 | 1406.60 | 1415.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 14:15:00 | 1409.20 | 1406.60 | 1415.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 1409.20 | 1406.60 | 1415.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 15:00:00 | 1409.20 | 1406.60 | 1415.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1399.95 | 1404.22 | 1413.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 1393.20 | 1400.40 | 1407.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 1378.05 | 1401.82 | 1407.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 15:15:00 | 1390.45 | 1381.80 | 1381.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 15:15:00 | 1390.45 | 1381.80 | 1381.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 1414.95 | 1388.43 | 1384.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1395.95 | 1407.27 | 1398.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1395.95 | 1407.27 | 1398.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1395.95 | 1407.27 | 1398.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 1395.95 | 1407.27 | 1398.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1395.10 | 1404.84 | 1398.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 1395.10 | 1404.84 | 1398.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 1424.25 | 1408.72 | 1400.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:15:00 | 1432.00 | 1408.72 | 1400.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 14:45:00 | 1427.85 | 1421.36 | 1408.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 12:30:00 | 1427.90 | 1425.07 | 1419.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 09:15:00 | 1429.55 | 1425.88 | 1421.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1438.50 | 1428.41 | 1423.16 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1393.70 | 1419.03 | 1422.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1393.70 | 1419.03 | 1422.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1390.10 | 1413.24 | 1419.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1400.00 | 1394.53 | 1406.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1400.00 | 1394.53 | 1406.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1400.00 | 1394.53 | 1406.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 1400.00 | 1394.53 | 1406.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1399.75 | 1395.57 | 1405.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1399.75 | 1395.57 | 1405.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1402.70 | 1395.31 | 1400.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 1402.70 | 1395.31 | 1400.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1407.55 | 1397.76 | 1401.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:15:00 | 1400.00 | 1398.45 | 1401.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 12:45:00 | 1400.00 | 1398.82 | 1401.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 1330.00 | 1345.72 | 1363.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 1330.00 | 1345.72 | 1363.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 15:15:00 | 1327.00 | 1318.48 | 1333.05 | SL hit (close>ema200) qty=0.50 sl=1318.48 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1247.45 | 1229.96 | 1228.52 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 1214.30 | 1228.83 | 1229.10 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 10:15:00 | 1240.00 | 1229.69 | 1229.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 1260.50 | 1236.33 | 1232.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1264.40 | 1272.82 | 1255.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 1264.40 | 1272.82 | 1255.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1264.40 | 1272.82 | 1255.55 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 1241.05 | 1248.18 | 1249.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 13:15:00 | 1235.65 | 1243.12 | 1246.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1252.00 | 1243.54 | 1245.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1252.00 | 1243.54 | 1245.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1252.00 | 1243.54 | 1245.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 1256.50 | 1243.54 | 1245.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1255.80 | 1245.99 | 1246.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 1258.20 | 1245.99 | 1246.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1200.05 | 1212.76 | 1223.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 1196.25 | 1212.76 | 1223.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 1199.05 | 1207.16 | 1217.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1136.44 | 1156.22 | 1173.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1139.10 | 1156.22 | 1173.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 15:15:00 | 1155.00 | 1148.26 | 1160.79 | SL hit (close>ema200) qty=0.50 sl=1148.26 alert=retest2 |

### Cycle 48 — BUY (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 13:15:00 | 1136.50 | 1124.98 | 1124.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 11:15:00 | 1146.75 | 1132.17 | 1128.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 13:15:00 | 1141.50 | 1145.01 | 1138.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 13:15:00 | 1141.50 | 1145.01 | 1138.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 1141.50 | 1145.01 | 1138.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 1140.00 | 1145.01 | 1138.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 1157.70 | 1147.55 | 1140.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 1143.15 | 1147.55 | 1140.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1128.10 | 1144.67 | 1140.58 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 12:15:00 | 1119.50 | 1134.58 | 1136.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 1108.60 | 1126.79 | 1132.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 899.90 | 890.49 | 933.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 899.90 | 890.49 | 933.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 897.05 | 891.92 | 906.33 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 922.90 | 910.20 | 909.91 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 904.80 | 912.24 | 912.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 887.65 | 903.60 | 907.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 902.10 | 900.76 | 904.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 14:00:00 | 902.10 | 900.76 | 904.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 888.00 | 897.06 | 901.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:00:00 | 883.00 | 894.25 | 900.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 882.70 | 888.15 | 894.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 885.10 | 888.92 | 893.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 14:15:00 | 892.00 | 878.95 | 877.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 892.00 | 878.95 | 877.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 914.40 | 888.13 | 882.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 891.90 | 905.38 | 896.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 891.90 | 905.38 | 896.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 891.90 | 905.38 | 896.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 935.95 | 922.75 | 913.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:45:00 | 935.35 | 926.83 | 916.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 916.75 | 923.69 | 924.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 916.75 | 923.69 | 924.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 910.35 | 921.02 | 923.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 906.85 | 906.78 | 913.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 10:30:00 | 909.95 | 906.78 | 913.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 914.05 | 906.29 | 910.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 914.05 | 906.29 | 910.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 920.00 | 909.03 | 911.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 928.30 | 909.03 | 911.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 925.00 | 912.22 | 912.68 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 936.20 | 917.02 | 914.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 11:15:00 | 946.95 | 923.00 | 917.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 918.25 | 934.57 | 926.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 09:15:00 | 918.25 | 934.57 | 926.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 918.25 | 934.57 | 926.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:45:00 | 940.05 | 932.74 | 929.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 11:45:00 | 939.50 | 937.20 | 933.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 13:00:00 | 938.20 | 937.40 | 934.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 14:45:00 | 938.05 | 937.78 | 934.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 936.45 | 937.52 | 935.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 924.90 | 937.52 | 935.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 915.20 | 933.05 | 933.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 915.20 | 933.05 | 933.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 872.15 | 912.08 | 921.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 894.50 | 890.81 | 903.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 894.50 | 890.81 | 903.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 894.50 | 890.81 | 903.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 887.15 | 895.93 | 900.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 907.55 | 898.70 | 898.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 907.55 | 898.70 | 898.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 911.75 | 901.31 | 899.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 936.30 | 937.58 | 927.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 14:45:00 | 936.35 | 937.58 | 927.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 942.75 | 943.02 | 937.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 949.25 | 943.02 | 937.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 944.00 | 967.72 | 968.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 944.00 | 967.72 | 968.23 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 14:15:00 | 972.05 | 968.60 | 968.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 10:15:00 | 976.20 | 971.31 | 969.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 13:15:00 | 971.45 | 971.69 | 970.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 13:15:00 | 971.45 | 971.69 | 970.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 971.45 | 971.69 | 970.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 971.45 | 971.69 | 970.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 966.75 | 970.70 | 970.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 966.75 | 970.70 | 970.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 968.40 | 970.24 | 969.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 976.55 | 970.24 | 969.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 972.80 | 972.34 | 971.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 12:00:00 | 972.80 | 972.34 | 971.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 12:15:00 | 973.00 | 972.47 | 971.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:00:00 | 973.00 | 972.47 | 971.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 972.20 | 972.84 | 971.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:30:00 | 971.25 | 972.84 | 971.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 984.00 | 975.07 | 972.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 967.60 | 975.07 | 972.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 990.60 | 978.17 | 974.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 10:45:00 | 1046.05 | 991.14 | 980.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 09:45:00 | 1027.90 | 1027.98 | 1007.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 1030.30 | 1027.98 | 1007.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 14:45:00 | 1027.70 | 1025.04 | 1013.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-05 09:15:00 | 1150.65 | 1055.65 | 1029.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 1304.70 | 1309.47 | 1310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 15:15:00 | 1294.90 | 1306.55 | 1308.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 1309.90 | 1306.96 | 1308.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 1309.90 | 1306.96 | 1308.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 1310.50 | 1307.67 | 1308.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:15:00 | 1311.00 | 1307.67 | 1308.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 1309.10 | 1307.95 | 1308.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 13:15:00 | 1303.80 | 1307.95 | 1308.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 1304.00 | 1293.70 | 1297.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 10:15:00 | 1300.70 | 1293.70 | 1297.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 1304.30 | 1298.30 | 1298.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 1301.10 | 1298.86 | 1298.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 14:15:00 | 1301.10 | 1298.86 | 1298.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 09:15:00 | 1323.00 | 1304.62 | 1301.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 1311.00 | 1313.08 | 1307.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 15:00:00 | 1311.00 | 1313.08 | 1307.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1307.00 | 1311.86 | 1307.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:15:00 | 1329.00 | 1311.86 | 1307.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 09:15:00 | 1326.00 | 1321.05 | 1315.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:30:00 | 1318.00 | 1316.68 | 1316.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 1298.20 | 1312.98 | 1314.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 1298.20 | 1312.98 | 1314.85 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 12:15:00 | 1319.60 | 1312.99 | 1312.98 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 13:15:00 | 1311.70 | 1312.73 | 1312.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 14:15:00 | 1309.90 | 1312.17 | 1312.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 1314.90 | 1312.71 | 1312.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 1326.60 | 1312.71 | 1312.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 1321.40 | 1314.45 | 1313.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 10:15:00 | 1337.70 | 1319.10 | 1315.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-02 09:15:00 | 1378.60 | 1381.47 | 1353.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-02 10:00:00 | 1378.60 | 1381.47 | 1353.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1375.00 | 1379.13 | 1367.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 1380.30 | 1376.57 | 1370.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:30:00 | 1383.20 | 1377.80 | 1371.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:00:00 | 1382.00 | 1378.64 | 1372.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 12:30:00 | 1382.10 | 1380.71 | 1374.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 15:15:00 | 1379.20 | 1382.21 | 1376.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1403.80 | 1382.21 | 1376.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 13:15:00 | 1387.50 | 1400.54 | 1400.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 13:15:00 | 1387.50 | 1400.54 | 1400.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 10:15:00 | 1381.10 | 1391.37 | 1396.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1383.20 | 1378.66 | 1386.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:30:00 | 1368.70 | 1377.15 | 1383.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 1367.20 | 1377.15 | 1383.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 09:30:00 | 1368.20 | 1374.14 | 1380.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:00:00 | 1369.00 | 1374.14 | 1380.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1377.80 | 1374.08 | 1379.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:45:00 | 1378.60 | 1374.08 | 1379.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1384.80 | 1376.23 | 1379.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:30:00 | 1388.90 | 1376.23 | 1379.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1364.20 | 1373.82 | 1378.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 1361.00 | 1373.03 | 1377.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 1361.00 | 1372.74 | 1375.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 1361.90 | 1370.28 | 1374.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 1360.00 | 1353.14 | 1359.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1356.80 | 1353.87 | 1359.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:00:00 | 1346.70 | 1352.44 | 1357.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1300.26 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1298.84 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1299.79 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1300.55 | 1321.14 | 1333.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 14:15:00 | 1325.40 | 1319.95 | 1330.68 | SL hit (close>ema200) qty=0.50 sl=1319.95 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 1350.50 | 1327.68 | 1326.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1383.50 | 1352.97 | 1346.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1358.80 | 1363.09 | 1355.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1358.80 | 1363.09 | 1355.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1356.00 | 1361.67 | 1355.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1356.00 | 1361.67 | 1355.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1347.90 | 1358.92 | 1354.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1352.30 | 1358.92 | 1354.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1346.00 | 1356.33 | 1353.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 12:00:00 | 1349.60 | 1354.99 | 1353.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:15:00 | 1349.90 | 1353.59 | 1352.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-30 14:15:00 | 1350.00 | 1352.39 | 1352.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-06-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 14:15:00 | 1350.00 | 1352.39 | 1352.42 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 09:15:00 | 1358.80 | 1353.12 | 1352.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 10:15:00 | 1368.80 | 1356.25 | 1354.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 12:15:00 | 1373.00 | 1374.78 | 1367.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 13:00:00 | 1373.00 | 1374.78 | 1367.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1366.40 | 1373.10 | 1367.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1368.30 | 1373.10 | 1367.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1362.80 | 1371.04 | 1367.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:45:00 | 1362.50 | 1371.04 | 1367.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 1362.00 | 1369.23 | 1366.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 1370.00 | 1369.23 | 1366.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 15:15:00 | 1372.00 | 1378.77 | 1379.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 15:15:00 | 1372.00 | 1378.77 | 1379.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 1348.40 | 1372.70 | 1376.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 1344.60 | 1343.81 | 1353.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 15:00:00 | 1344.60 | 1343.81 | 1353.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1351.60 | 1345.37 | 1353.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 1315.90 | 1345.37 | 1353.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 1335.00 | 1343.13 | 1348.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 1335.70 | 1343.11 | 1346.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:45:00 | 1335.00 | 1340.52 | 1342.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 1371.70 | 1342.69 | 1342.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 1371.70 | 1342.69 | 1342.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1405.30 | 1381.56 | 1369.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 1380.00 | 1381.25 | 1370.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:30:00 | 1380.00 | 1381.25 | 1370.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1439.60 | 1462.23 | 1444.72 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 1430.40 | 1437.81 | 1438.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 1428.00 | 1434.68 | 1437.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1458.40 | 1437.39 | 1437.50 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 1457.00 | 1441.31 | 1439.27 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1436.50 | 1446.93 | 1447.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 14:15:00 | 1403.90 | 1433.33 | 1440.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1382.10 | 1380.23 | 1397.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 1382.10 | 1380.23 | 1397.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1387.40 | 1382.57 | 1395.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 1387.10 | 1382.57 | 1395.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1392.80 | 1386.76 | 1394.42 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1412.90 | 1399.65 | 1398.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 1428.30 | 1410.23 | 1403.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1421.90 | 1430.91 | 1420.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:00:00 | 1421.90 | 1430.91 | 1420.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 1421.90 | 1429.11 | 1420.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:45:00 | 1425.20 | 1429.11 | 1420.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 1414.40 | 1426.17 | 1419.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 1414.40 | 1426.17 | 1419.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 1375.80 | 1416.10 | 1415.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 1375.80 | 1416.10 | 1415.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 1334.60 | 1399.80 | 1408.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 1307.40 | 1371.27 | 1393.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 14:15:00 | 1258.00 | 1256.49 | 1283.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 1258.00 | 1256.49 | 1283.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1281.00 | 1259.04 | 1271.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1281.00 | 1259.04 | 1271.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1284.00 | 1264.03 | 1272.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1268.30 | 1264.03 | 1272.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 1258.30 | 1262.00 | 1270.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 1254.20 | 1260.44 | 1268.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 1253.90 | 1259.13 | 1267.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 09:15:00 | 1226.00 | 1219.24 | 1219.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1226.00 | 1219.24 | 1219.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1228.50 | 1221.64 | 1220.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1223.20 | 1223.30 | 1221.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 1223.70 | 1223.30 | 1221.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1220.30 | 1222.70 | 1221.51 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1210.00 | 1220.16 | 1220.46 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-22 09:15:00 | 1236.90 | 1222.04 | 1221.17 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1219.00 | 1221.01 | 1221.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 1213.50 | 1219.50 | 1220.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1219.80 | 1219.16 | 1220.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:15:00 | 1215.80 | 1219.26 | 1219.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1202.80 | 1190.75 | 1190.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1202.80 | 1190.75 | 1190.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 1209.30 | 1195.63 | 1193.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1210.60 | 1215.84 | 1210.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1210.10 | 1215.84 | 1210.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1206.70 | 1214.01 | 1210.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 1206.70 | 1214.01 | 1210.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1212.50 | 1213.71 | 1210.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 13:45:00 | 1213.80 | 1213.97 | 1210.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 1205.00 | 1209.01 | 1209.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 1205.00 | 1209.01 | 1209.29 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1219.20 | 1211.00 | 1210.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1228.00 | 1220.07 | 1215.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 11:15:00 | 1218.50 | 1220.93 | 1217.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 12:00:00 | 1218.50 | 1220.93 | 1217.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1214.40 | 1219.63 | 1216.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1214.40 | 1219.63 | 1216.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1214.00 | 1218.50 | 1216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:30:00 | 1205.50 | 1218.50 | 1216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 14:15:00 | 1220.50 | 1218.90 | 1216.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1229.90 | 1219.12 | 1217.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 12:15:00 | 1242.40 | 1247.00 | 1247.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 12:15:00 | 1242.40 | 1247.00 | 1247.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 13:15:00 | 1239.00 | 1245.40 | 1246.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1243.60 | 1240.89 | 1243.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 1243.60 | 1240.89 | 1243.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1247.00 | 1242.11 | 1243.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 1247.30 | 1242.11 | 1243.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1248.00 | 1243.29 | 1244.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 1248.00 | 1243.29 | 1244.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 15:15:00 | 1250.20 | 1245.76 | 1245.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 09:15:00 | 1256.80 | 1247.97 | 1246.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1267.10 | 1267.40 | 1260.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:30:00 | 1265.10 | 1267.40 | 1260.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1275.20 | 1288.78 | 1279.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1275.20 | 1288.78 | 1279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1273.50 | 1285.72 | 1279.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1273.50 | 1285.72 | 1279.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 1255.80 | 1274.36 | 1275.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1247.00 | 1260.86 | 1266.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1233.90 | 1232.44 | 1246.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:45:00 | 1234.70 | 1232.44 | 1246.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1230.20 | 1231.95 | 1240.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:30:00 | 1238.60 | 1231.95 | 1240.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1240.00 | 1233.56 | 1240.74 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 1254.90 | 1244.82 | 1244.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 1260.50 | 1249.58 | 1246.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1236.20 | 1247.82 | 1246.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 1236.20 | 1247.82 | 1246.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 1240.40 | 1246.34 | 1245.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 1236.00 | 1246.34 | 1245.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1258.40 | 1248.46 | 1246.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-01 14:30:00 | 1265.40 | 1254.17 | 1249.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 12:30:00 | 1266.50 | 1263.83 | 1257.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1269.60 | 1260.20 | 1256.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:15:00 | 1266.20 | 1263.61 | 1259.54 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1257.80 | 1263.66 | 1260.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 1270.00 | 1263.26 | 1261.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 1269.00 | 1264.19 | 1262.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:45:00 | 1270.00 | 1265.21 | 1263.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 1255.00 | 1262.96 | 1262.71 | SL hit (close<static) qty=1.00 sl=1256.70 alert=retest2 |

### Cycle 87 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 1254.40 | 1261.25 | 1261.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 1251.10 | 1259.22 | 1260.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1256.10 | 1254.62 | 1257.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 1256.10 | 1254.62 | 1257.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1268.00 | 1257.74 | 1258.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 1281.00 | 1257.74 | 1258.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 1270.90 | 1260.37 | 1259.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 1283.60 | 1266.88 | 1262.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 1261.60 | 1268.63 | 1265.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 1261.60 | 1268.63 | 1265.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1269.80 | 1268.86 | 1265.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 1265.00 | 1268.86 | 1265.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 1265.90 | 1268.27 | 1265.97 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 15:15:00 | 1258.00 | 1264.33 | 1264.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 1257.00 | 1262.86 | 1263.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1261.90 | 1259.30 | 1261.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 1255.70 | 1259.30 | 1261.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1248.60 | 1257.16 | 1260.00 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1272.20 | 1260.44 | 1260.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 1278.00 | 1268.68 | 1264.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1270.00 | 1272.10 | 1268.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:45:00 | 1270.90 | 1272.10 | 1268.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1262.70 | 1270.22 | 1267.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 1262.70 | 1270.22 | 1267.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1261.20 | 1268.41 | 1267.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 1271.40 | 1267.13 | 1266.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 1254.30 | 1264.56 | 1265.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1254.30 | 1264.56 | 1265.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 14:15:00 | 1253.10 | 1257.87 | 1261.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 1261.50 | 1257.66 | 1260.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1264.40 | 1257.66 | 1260.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1258.00 | 1257.73 | 1260.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1258.00 | 1257.73 | 1260.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1264.50 | 1259.08 | 1260.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:15:00 | 1256.30 | 1259.34 | 1260.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 1252.60 | 1258.85 | 1260.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 1335.90 | 1265.96 | 1260.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1335.90 | 1265.96 | 1260.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 1355.20 | 1316.99 | 1293.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 11:15:00 | 1409.80 | 1410.12 | 1382.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 12:00:00 | 1409.80 | 1410.12 | 1382.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1406.00 | 1413.22 | 1405.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:30:00 | 1404.80 | 1413.22 | 1405.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 13:15:00 | 1406.70 | 1411.91 | 1405.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1443.80 | 1403.21 | 1402.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 10:30:00 | 1428.10 | 1414.82 | 1408.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:00:00 | 1432.40 | 1414.82 | 1408.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 1427.40 | 1418.54 | 1410.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 1420.80 | 1420.58 | 1413.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 1414.70 | 1420.58 | 1413.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1399.00 | 1416.17 | 1412.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1399.00 | 1416.17 | 1412.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1393.00 | 1411.53 | 1411.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1391.20 | 1411.53 | 1411.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1390.00 | 1407.23 | 1409.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 1390.00 | 1407.23 | 1409.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1356.30 | 1393.04 | 1401.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 13:15:00 | 1358.20 | 1353.75 | 1364.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 14:00:00 | 1358.20 | 1353.75 | 1364.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1388.00 | 1361.23 | 1365.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:15:00 | 1392.20 | 1361.23 | 1365.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1394.90 | 1367.97 | 1367.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 1403.90 | 1382.06 | 1374.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1390.90 | 1393.27 | 1384.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 13:00:00 | 1390.90 | 1393.27 | 1384.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1372.30 | 1388.22 | 1384.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1372.30 | 1388.22 | 1384.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1365.00 | 1383.57 | 1382.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 1395.50 | 1383.57 | 1382.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:45:00 | 1374.80 | 1382.25 | 1381.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 1378.60 | 1381.52 | 1381.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1378.60 | 1381.52 | 1381.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 1376.50 | 1380.51 | 1381.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1386.70 | 1378.13 | 1379.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 1386.70 | 1378.13 | 1379.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 1384.60 | 1379.42 | 1379.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 1382.40 | 1379.42 | 1379.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 1383.20 | 1380.61 | 1380.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 1383.20 | 1380.61 | 1380.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1389.50 | 1382.39 | 1381.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 1381.60 | 1382.77 | 1381.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 1361.20 | 1382.77 | 1381.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1353.70 | 1376.96 | 1379.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 13:15:00 | 1349.20 | 1363.20 | 1371.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1363.70 | 1355.37 | 1363.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 11:45:00 | 1363.50 | 1355.37 | 1363.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1360.00 | 1356.30 | 1363.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:15:00 | 1353.90 | 1359.23 | 1362.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 1385.50 | 1364.53 | 1364.72 | SL hit (close>static) qty=1.00 sl=1363.70 alert=retest2 |

### Cycle 98 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1379.60 | 1367.54 | 1366.07 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1355.50 | 1366.59 | 1366.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 1344.00 | 1360.14 | 1363.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1346.70 | 1341.72 | 1350.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1346.70 | 1341.72 | 1350.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1347.50 | 1342.87 | 1350.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1339.00 | 1342.87 | 1350.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1352.00 | 1344.70 | 1350.37 | SL hit (close>static) qty=1.00 sl=1350.70 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1365.40 | 1354.09 | 1353.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1409.00 | 1364.58 | 1357.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 1387.10 | 1387.87 | 1375.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:15:00 | 1387.40 | 1387.87 | 1375.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1382.00 | 1386.32 | 1379.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1379.00 | 1386.32 | 1379.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1391.60 | 1401.71 | 1395.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:00:00 | 1391.60 | 1401.71 | 1395.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1391.80 | 1399.73 | 1395.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 1397.90 | 1399.73 | 1395.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1391.40 | 1398.06 | 1395.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1391.80 | 1398.06 | 1395.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 1401.00 | 1399.39 | 1396.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 1396.50 | 1399.39 | 1396.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1404.50 | 1400.41 | 1397.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 1388.10 | 1400.41 | 1397.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1397.00 | 1400.08 | 1397.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 1397.00 | 1400.08 | 1397.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 12:15:00 | 1394.80 | 1399.02 | 1397.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:45:00 | 1395.40 | 1399.02 | 1397.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 13:15:00 | 1399.70 | 1399.16 | 1397.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1423.10 | 1398.72 | 1397.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 1400.00 | 1403.27 | 1400.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 13:45:00 | 1399.80 | 1402.80 | 1400.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1392.60 | 1400.76 | 1399.91 | SL hit (close<static) qty=1.00 sl=1393.50 alert=retest2 |

### Cycle 101 — SELL (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 09:15:00 | 1389.60 | 1397.93 | 1398.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1373.30 | 1390.78 | 1394.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 10:15:00 | 1369.90 | 1367.66 | 1377.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 1369.90 | 1367.66 | 1377.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 1374.10 | 1365.43 | 1371.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:00:00 | 1374.10 | 1365.43 | 1371.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1382.60 | 1368.86 | 1372.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1382.60 | 1368.86 | 1372.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 1392.00 | 1377.07 | 1375.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1408.00 | 1385.29 | 1379.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 1409.00 | 1409.01 | 1396.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 1409.70 | 1409.01 | 1396.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1438.90 | 1448.54 | 1440.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 1438.90 | 1448.54 | 1440.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1449.00 | 1448.63 | 1441.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1430.80 | 1448.63 | 1441.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1424.50 | 1443.80 | 1439.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1428.40 | 1443.80 | 1439.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1427.30 | 1437.25 | 1437.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1421.90 | 1434.18 | 1436.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1437.10 | 1434.10 | 1435.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1414.40 | 1434.10 | 1435.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1464.80 | 1435.81 | 1433.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1464.80 | 1435.81 | 1433.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 10:15:00 | 1502.70 | 1449.19 | 1439.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 1523.40 | 1525.64 | 1500.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:30:00 | 1521.80 | 1525.64 | 1500.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 1513.70 | 1524.02 | 1513.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 1513.70 | 1524.02 | 1513.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1509.00 | 1521.02 | 1513.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:00:00 | 1509.00 | 1521.02 | 1513.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1510.60 | 1518.93 | 1512.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 1508.50 | 1518.93 | 1512.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1510.00 | 1517.15 | 1512.53 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1493.60 | 1509.77 | 1509.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1486.60 | 1505.14 | 1507.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 09:15:00 | 1458.10 | 1457.06 | 1472.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 09:45:00 | 1460.90 | 1457.06 | 1472.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1455.00 | 1456.65 | 1471.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 1450.00 | 1455.32 | 1469.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 1449.60 | 1447.69 | 1455.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 1451.30 | 1448.81 | 1455.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1442.90 | 1449.83 | 1454.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1456.00 | 1451.07 | 1455.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1462.10 | 1451.07 | 1455.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1458.20 | 1452.49 | 1455.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:30:00 | 1456.00 | 1452.49 | 1455.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1465.80 | 1455.15 | 1456.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1465.80 | 1455.15 | 1456.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-01 12:15:00 | 1467.70 | 1457.66 | 1457.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1467.70 | 1457.66 | 1457.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 1470.00 | 1462.26 | 1459.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 1531.00 | 1534.30 | 1521.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 1534.30 | 1534.30 | 1521.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1557.00 | 1541.74 | 1532.67 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 1515.50 | 1530.91 | 1531.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1482.20 | 1509.86 | 1519.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1514.10 | 1495.76 | 1504.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 1514.10 | 1495.76 | 1504.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 1506.10 | 1497.82 | 1504.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1498.90 | 1499.49 | 1504.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 1498.20 | 1500.19 | 1504.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:30:00 | 1497.90 | 1498.65 | 1501.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1498.00 | 1498.90 | 1501.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1498.00 | 1498.72 | 1500.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 1481.00 | 1498.72 | 1500.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1505.40 | 1500.06 | 1501.37 | SL hit (close>static) qty=1.00 sl=1501.00 alert=retest2 |

### Cycle 108 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 1373.50 | 1345.94 | 1344.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1426.80 | 1367.55 | 1355.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1381.20 | 1381.48 | 1365.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1376.30 | 1381.48 | 1365.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1372.00 | 1379.49 | 1368.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 1353.80 | 1379.49 | 1368.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1342.50 | 1372.09 | 1366.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:15:00 | 1319.90 | 1372.09 | 1366.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 1330.00 | 1363.67 | 1363.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 1330.00 | 1363.67 | 1363.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2026-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 11:15:00 | 1329.60 | 1356.86 | 1360.09 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1417.90 | 1368.60 | 1363.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 1491.50 | 1437.35 | 1407.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 1450.00 | 1453.05 | 1428.21 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 09:15:00 | 1465.40 | 1453.05 | 1428.21 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1452.20 | 1466.18 | 1450.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1452.20 | 1466.18 | 1450.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1460.00 | 1464.94 | 1451.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:00:00 | 1462.60 | 1462.76 | 1453.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:45:00 | 1468.00 | 1463.81 | 1456.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 12:00:00 | 1462.30 | 1463.55 | 1457.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 1465.30 | 1463.90 | 1458.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1447.00 | 1461.42 | 1459.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 1447.00 | 1461.42 | 1459.09 | SL hit (close<ema400) qty=1.00 sl=1459.09 alert=retest1 |

### Cycle 111 — SELL (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 11:15:00 | 1448.90 | 1456.56 | 1457.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 12:15:00 | 1442.60 | 1453.77 | 1455.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1452.90 | 1440.97 | 1447.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1452.90 | 1440.97 | 1447.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1458.00 | 1444.38 | 1448.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 1458.00 | 1444.38 | 1448.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1416.90 | 1412.76 | 1417.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1418.40 | 1412.76 | 1417.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1411.30 | 1412.47 | 1417.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1415.60 | 1412.47 | 1417.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1415.70 | 1413.12 | 1417.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1415.70 | 1413.12 | 1417.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1412.70 | 1413.03 | 1416.67 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1423.30 | 1419.27 | 1418.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1442.00 | 1425.87 | 1422.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1421.20 | 1430.60 | 1425.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1421.20 | 1430.60 | 1425.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 1418.80 | 1428.24 | 1425.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 09:45:00 | 1432.90 | 1429.81 | 1426.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 1408.00 | 1426.52 | 1426.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1408.00 | 1426.52 | 1426.78 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 1444.00 | 1426.40 | 1426.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1458.10 | 1437.41 | 1431.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 1432.60 | 1436.45 | 1431.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 1432.60 | 1436.45 | 1431.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1443.50 | 1437.86 | 1432.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 1443.50 | 1437.86 | 1432.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1492.00 | 1455.15 | 1443.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 09:15:00 | 1525.00 | 1484.01 | 1464.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-25 12:00:00 | 1516.10 | 1504.03 | 1479.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 1550.10 | 1503.72 | 1487.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 13:00:00 | 1518.10 | 1513.12 | 1497.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 1556.00 | 1551.88 | 1532.91 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 1496.00 | 1524.21 | 1526.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 1496.00 | 1524.21 | 1526.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 1487.30 | 1516.83 | 1523.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1498.10 | 1481.96 | 1495.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 1498.10 | 1481.96 | 1495.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1498.00 | 1485.17 | 1495.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 1506.00 | 1485.17 | 1495.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1532.70 | 1494.67 | 1498.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 1532.70 | 1494.67 | 1498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1534.00 | 1502.54 | 1502.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 1548.70 | 1517.49 | 1509.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1493.30 | 1518.56 | 1513.30 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 1498.50 | 1509.94 | 1510.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-10 09:15:00 | 1478.90 | 1497.56 | 1503.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 1433.40 | 1430.97 | 1450.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 1433.40 | 1430.97 | 1450.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1388.70 | 1370.40 | 1384.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1388.70 | 1370.40 | 1384.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1376.80 | 1371.68 | 1383.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1372.10 | 1371.68 | 1383.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 1375.10 | 1374.07 | 1381.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 1392.80 | 1377.82 | 1382.70 | SL hit (close>static) qty=1.00 sl=1390.00 alert=retest2 |

### Cycle 118 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1414.00 | 1387.64 | 1386.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 1428.00 | 1404.28 | 1395.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1416.80 | 1419.73 | 1406.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 09:45:00 | 1411.80 | 1419.73 | 1406.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1395.70 | 1412.64 | 1407.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 14:00:00 | 1395.70 | 1412.64 | 1407.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1394.90 | 1409.09 | 1406.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1394.90 | 1409.09 | 1406.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 10:15:00 | 1393.10 | 1403.21 | 1403.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1332.60 | 1383.94 | 1394.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 1352.90 | 1335.99 | 1353.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 1352.90 | 1335.99 | 1353.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1353.10 | 1339.41 | 1353.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1353.10 | 1339.41 | 1353.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 1350.10 | 1341.55 | 1353.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 1358.10 | 1341.55 | 1353.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 1355.00 | 1344.24 | 1353.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 1379.00 | 1344.24 | 1353.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1368.60 | 1349.11 | 1354.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 1369.00 | 1349.11 | 1354.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 1359.10 | 1355.97 | 1356.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:30:00 | 1358.00 | 1355.97 | 1356.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 1352.10 | 1355.19 | 1356.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:45:00 | 1349.30 | 1353.62 | 1355.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1338.00 | 1323.41 | 1323.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1338.00 | 1323.41 | 1323.08 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1299.00 | 1320.18 | 1321.89 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 1333.00 | 1321.98 | 1320.89 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 10:15:00 | 1314.20 | 1318.99 | 1319.63 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 1326.10 | 1320.45 | 1320.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 1327.40 | 1321.84 | 1320.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1323.70 | 1325.06 | 1322.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1345.00 | 1323.76 | 1323.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 15:15:00 | 1375.10 | 1380.66 | 1381.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1375.10 | 1380.66 | 1381.24 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 1396.70 | 1383.87 | 1382.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 12:15:00 | 1407.90 | 1394.14 | 1389.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1471.10 | 1478.53 | 1458.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 14:00:00 | 1471.10 | 1478.53 | 1458.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1460.70 | 1472.87 | 1461.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 1461.40 | 1472.87 | 1461.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 1460.20 | 1470.33 | 1460.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:45:00 | 1460.00 | 1470.33 | 1460.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 1473.50 | 1470.97 | 1462.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:30:00 | 1456.50 | 1470.97 | 1462.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1459.00 | 1467.69 | 1462.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:45:00 | 1455.60 | 1467.69 | 1462.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1458.80 | 1465.91 | 1462.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 1450.80 | 1465.91 | 1462.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 1463.00 | 1468.24 | 1465.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:30:00 | 1463.00 | 1468.24 | 1465.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1460.00 | 1466.59 | 1465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1431.80 | 1466.59 | 1465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1435.20 | 1460.31 | 1462.40 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1487.00 | 1451.98 | 1447.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 11:15:00 | 1494.30 | 1460.44 | 1451.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 11:15:00 | 1598.80 | 1608.62 | 1581.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 12:00:00 | 1598.80 | 1608.62 | 1581.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 1580.00 | 1603.48 | 1586.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 1534.80 | 1603.48 | 1586.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1571.00 | 1596.98 | 1584.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 1633.10 | 1596.98 | 1584.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-04 14:15:00 | 1796.41 | 1700.40 | 1647.57 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 14:45:00 | 1746.20 | 2024-05-23 15:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-05-22 09:15:00 | 1757.00 | 2024-05-23 15:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-05-22 09:45:00 | 1755.55 | 2024-05-23 15:15:00 | 1742.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-05-23 09:15:00 | 1774.75 | 2024-05-23 15:15:00 | 1742.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-06-03 11:30:00 | 1708.40 | 2024-06-04 10:15:00 | 1622.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1658.75 | 2024-06-04 10:15:00 | 1575.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:30:00 | 1708.40 | 2024-06-04 11:15:00 | 1537.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 09:15:00 | 1658.75 | 2024-06-04 11:15:00 | 1492.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1763.65 | 2024-06-20 09:15:00 | 1750.95 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-06-12 14:30:00 | 1742.65 | 2024-06-20 09:15:00 | 1750.95 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-06-24 09:15:00 | 1725.50 | 2024-06-24 14:15:00 | 1756.70 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-06-24 15:15:00 | 1749.00 | 2024-06-25 09:15:00 | 1759.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-05 09:15:00 | 1816.45 | 2024-07-11 12:15:00 | 1809.65 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-07-05 10:30:00 | 1800.05 | 2024-07-11 12:15:00 | 1809.65 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2024-07-08 11:15:00 | 1816.10 | 2024-07-11 12:15:00 | 1809.65 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-07-10 11:00:00 | 1799.50 | 2024-07-11 12:15:00 | 1809.65 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2024-07-16 14:30:00 | 1771.00 | 2024-07-18 09:15:00 | 1787.95 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-07-16 15:00:00 | 1770.65 | 2024-07-18 09:15:00 | 1787.95 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-18 12:45:00 | 1771.70 | 2024-07-18 15:15:00 | 1787.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-07-19 09:15:00 | 1757.70 | 2024-07-24 15:15:00 | 1739.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2024-07-19 11:45:00 | 1741.35 | 2024-07-24 15:15:00 | 1739.00 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2024-07-19 12:15:00 | 1731.65 | 2024-07-24 15:15:00 | 1739.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-07-19 14:45:00 | 1740.00 | 2024-07-24 15:15:00 | 1739.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-07-24 10:00:00 | 1741.35 | 2024-07-24 15:15:00 | 1739.00 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1804.40 | 2024-08-01 09:15:00 | 1728.95 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2024-07-31 10:15:00 | 1779.10 | 2024-08-01 09:15:00 | 1728.95 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-08-21 15:15:00 | 1652.00 | 2024-08-22 11:15:00 | 1639.45 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-08-28 13:45:00 | 1606.85 | 2024-08-30 15:15:00 | 1615.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-30 13:30:00 | 1607.20 | 2024-08-30 15:15:00 | 1615.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-09-02 14:15:00 | 1624.45 | 2024-09-03 09:15:00 | 1604.05 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-09-04 10:30:00 | 1596.80 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2024-09-04 11:15:00 | 1595.00 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2024-09-04 14:15:00 | 1588.30 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2024-09-04 15:00:00 | 1591.85 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2024-09-05 10:15:00 | 1585.00 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2024-09-05 12:00:00 | 1583.45 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-09-06 10:15:00 | 1585.00 | 2024-09-10 15:15:00 | 1578.00 | STOP_HIT | 1.00 | 0.44% |
| BUY | retest2 | 2024-09-17 10:45:00 | 1660.10 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2024-09-17 12:45:00 | 1660.05 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 1.09% |
| BUY | retest2 | 2024-09-18 10:00:00 | 1658.85 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2024-09-18 13:15:00 | 1663.25 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-09-19 14:00:00 | 1671.00 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-09-20 10:30:00 | 1670.95 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 0.43% |
| BUY | retest2 | 2024-09-20 11:00:00 | 1672.20 | 2024-09-25 11:15:00 | 1678.15 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-10-03 10:15:00 | 1726.75 | 2024-10-07 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-10-03 15:15:00 | 1723.50 | 2024-10-07 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-10-04 09:45:00 | 1723.25 | 2024-10-07 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-10-04 10:30:00 | 1732.00 | 2024-10-07 10:15:00 | 1706.90 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-11 11:45:00 | 1725.00 | 2024-10-14 12:15:00 | 1740.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-10-16 15:00:00 | 1786.00 | 2024-10-17 13:15:00 | 1752.55 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-10-21 11:30:00 | 1706.50 | 2024-10-23 09:15:00 | 1621.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:30:00 | 1706.50 | 2024-10-25 09:15:00 | 1535.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-05 09:15:00 | 1584.15 | 2024-11-11 11:15:00 | 1506.85 | STOP_HIT | 1.00 | -4.88% |
| BUY | retest2 | 2024-11-05 15:15:00 | 1542.70 | 2024-11-11 11:15:00 | 1506.85 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-11-06 12:00:00 | 1525.15 | 2024-11-11 11:15:00 | 1506.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-11-12 11:45:00 | 1507.15 | 2024-11-13 15:15:00 | 1537.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-11-27 13:30:00 | 1465.30 | 2024-12-02 13:15:00 | 1467.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2024-11-27 14:15:00 | 1465.00 | 2024-12-02 13:15:00 | 1467.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-11-28 11:15:00 | 1464.70 | 2024-12-02 13:15:00 | 1467.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-11-28 12:00:00 | 1463.80 | 2024-12-02 13:15:00 | 1467.00 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-12-06 09:15:00 | 1499.00 | 2024-12-06 15:15:00 | 1479.80 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest1 | 2024-12-06 11:15:00 | 1491.50 | 2024-12-06 15:15:00 | 1479.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-12-20 15:00:00 | 1393.20 | 2024-12-27 15:15:00 | 1390.45 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-12-23 09:15:00 | 1378.05 | 2024-12-27 15:15:00 | 1390.45 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-31 12:15:00 | 1432.00 | 2025-01-06 10:15:00 | 1393.70 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-12-31 14:45:00 | 1427.85 | 2025-01-06 10:15:00 | 1393.70 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-01-02 12:30:00 | 1427.90 | 2025-01-06 10:15:00 | 1393.70 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-01-03 09:15:00 | 1429.55 | 2025-01-06 10:15:00 | 1393.70 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-01-08 12:15:00 | 1400.00 | 2025-01-13 12:15:00 | 1330.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:45:00 | 1400.00 | 2025-01-13 12:15:00 | 1330.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 12:15:00 | 1400.00 | 2025-01-14 15:15:00 | 1327.00 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-01-08 12:45:00 | 1400.00 | 2025-01-14 15:15:00 | 1327.00 | STOP_HIT | 0.50 | 5.21% |
| SELL | retest2 | 2025-02-07 10:15:00 | 1196.25 | 2025-02-12 09:15:00 | 1136.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1199.05 | 2025-02-12 09:15:00 | 1139.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 10:15:00 | 1196.25 | 2025-02-12 15:15:00 | 1155.00 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2025-02-07 12:45:00 | 1199.05 | 2025-02-12 15:15:00 | 1155.00 | STOP_HIT | 0.50 | 3.67% |
| SELL | retest2 | 2025-03-12 11:00:00 | 883.00 | 2025-03-18 14:15:00 | 892.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-03-13 09:15:00 | 882.70 | 2025-03-18 14:15:00 | 892.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-13 11:15:00 | 885.10 | 2025-03-18 14:15:00 | 892.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-03-24 09:15:00 | 935.95 | 2025-03-25 15:15:00 | 916.75 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-03-24 09:45:00 | 935.35 | 2025-03-25 15:15:00 | 916.75 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-04-02 11:45:00 | 940.05 | 2025-04-04 09:15:00 | 915.20 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-04-03 11:45:00 | 939.50 | 2025-04-04 09:15:00 | 915.20 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-04-03 13:00:00 | 938.20 | 2025-04-04 09:15:00 | 915.20 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-04-03 14:45:00 | 938.05 | 2025-04-04 09:15:00 | 915.20 | STOP_HIT | 1.00 | -2.44% |
| SELL | retest2 | 2025-04-09 10:00:00 | 887.15 | 2025-04-11 11:15:00 | 907.55 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-04-21 10:15:00 | 949.25 | 2025-04-25 10:15:00 | 944.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2025-04-30 10:45:00 | 1046.05 | 2025-05-05 09:15:00 | 1150.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 09:45:00 | 1027.90 | 2025-05-05 09:15:00 | 1130.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 10:15:00 | 1030.30 | 2025-05-05 09:15:00 | 1133.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-02 14:45:00 | 1027.70 | 2025-05-05 09:15:00 | 1130.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 11:00:00 | 1231.80 | 2025-05-16 09:15:00 | 1354.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 11:30:00 | 1232.00 | 2025-05-16 09:15:00 | 1355.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 14:45:00 | 1231.10 | 2025-05-16 09:15:00 | 1354.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1297.80 | 2025-05-19 14:15:00 | 1304.70 | STOP_HIT | 1.00 | 0.53% |
| SELL | retest2 | 2025-05-20 13:15:00 | 1303.80 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-05-22 09:30:00 | 1304.00 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-05-22 10:15:00 | 1300.70 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-05-22 13:30:00 | 1304.30 | 2025-05-22 14:15:00 | 1301.10 | STOP_HIT | 1.00 | 0.25% |
| BUY | retest2 | 2025-05-26 09:15:00 | 1329.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-05-27 09:15:00 | 1326.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-28 09:30:00 | 1318.00 | 2025-05-28 10:15:00 | 1298.20 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-04 10:00:00 | 1380.30 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-06-04 10:30:00 | 1383.20 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2025-06-04 12:00:00 | 1382.00 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.40% |
| BUY | retest2 | 2025-06-04 12:30:00 | 1382.10 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1403.80 | 2025-06-09 13:15:00 | 1387.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1368.70 | 2025-06-19 12:15:00 | 1300.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 14:00:00 | 1367.20 | 2025-06-19 12:15:00 | 1298.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1368.20 | 2025-06-19 12:15:00 | 1299.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1369.00 | 2025-06-19 12:15:00 | 1300.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 13:30:00 | 1368.70 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.16% |
| SELL | retest2 | 2025-06-11 14:00:00 | 1367.20 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.06% |
| SELL | retest2 | 2025-06-12 09:30:00 | 1368.20 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-06-12 10:00:00 | 1369.00 | 2025-06-19 14:15:00 | 1325.40 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1292.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 09:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-06-13 12:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1292.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 12:15:00 | 1361.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2025-06-13 12:45:00 | 1361.90 | 2025-06-23 09:15:00 | 1293.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 1361.90 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-06-17 09:45:00 | 1360.00 | 2025-06-23 09:15:00 | 1292.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:45:00 | 1360.00 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2025-06-17 12:00:00 | 1346.70 | 2025-06-23 09:15:00 | 1279.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 12:00:00 | 1346.70 | 2025-06-23 09:15:00 | 1325.80 | STOP_HIT | 0.50 | 1.55% |
| BUY | retest2 | 2025-06-30 12:00:00 | 1349.60 | 2025-06-30 14:15:00 | 1350.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-06-30 13:15:00 | 1349.90 | 2025-06-30 14:15:00 | 1350.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-07-03 09:15:00 | 1370.00 | 2025-07-04 15:15:00 | 1372.00 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-07-09 09:15:00 | 1315.90 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -4.24% |
| SELL | retest2 | 2025-07-09 14:15:00 | 1335.00 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-07-10 11:15:00 | 1335.70 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-07-11 11:45:00 | 1335.00 | 2025-07-14 09:15:00 | 1371.70 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-08-08 12:00:00 | 1254.20 | 2025-08-20 09:15:00 | 1226.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest2 | 2025-08-08 13:00:00 | 1253.90 | 2025-08-20 09:15:00 | 1226.00 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2025-08-25 15:15:00 | 1215.80 | 2025-09-02 09:15:00 | 1202.80 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-09-05 13:45:00 | 1213.80 | 2025-09-08 11:15:00 | 1205.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1229.90 | 2025-09-17 12:15:00 | 1242.40 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2025-10-01 14:30:00 | 1265.40 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-03 12:30:00 | 1266.50 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-06 09:30:00 | 1269.60 | 2025-10-08 14:15:00 | 1255.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-06 13:15:00 | 1266.20 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-07 14:15:00 | 1270.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-08 09:30:00 | 1269.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-10-08 11:45:00 | 1270.00 | 2025-10-08 15:15:00 | 1254.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-10-20 09:15:00 | 1271.40 | 2025-10-20 09:15:00 | 1254.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-10-23 12:15:00 | 1256.30 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.34% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1252.60 | 2025-10-27 09:15:00 | 1335.90 | STOP_HIT | 1.00 | -6.65% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1443.80 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-11-04 10:30:00 | 1428.10 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-11-04 11:00:00 | 1432.40 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-11-04 11:30:00 | 1427.40 | 2025-11-06 11:15:00 | 1390.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-11-14 09:15:00 | 1395.50 | 2025-11-14 11:15:00 | 1378.60 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-11-14 10:45:00 | 1374.80 | 2025-11-14 11:15:00 | 1378.60 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-11-17 11:15:00 | 1382.40 | 2025-11-17 12:15:00 | 1383.20 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-11-20 09:15:00 | 1353.90 | 2025-11-20 10:15:00 | 1385.50 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1339.00 | 2025-11-25 09:15:00 | 1352.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-25 11:00:00 | 1344.40 | 2025-11-25 11:15:00 | 1352.90 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-12-03 09:15:00 | 1423.10 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-12-03 12:30:00 | 1400.00 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-12-03 13:45:00 | 1399.80 | 2025-12-03 14:15:00 | 1392.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1414.40 | 2025-12-19 09:15:00 | 1464.80 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-12-30 12:00:00 | 1450.00 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-31 13:15:00 | 1449.60 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-12-31 15:00:00 | 1451.30 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1442.90 | 2026-01-01 12:15:00 | 1467.70 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1498.90 | 2026-01-16 09:15:00 | 1505.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-13 15:15:00 | 1498.20 | 2026-01-19 11:15:00 | 1423.95 | PARTIAL | 0.50 | 4.96% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1497.90 | 2026-01-19 11:15:00 | 1423.29 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1498.00 | 2026-01-19 11:15:00 | 1423.01 | PARTIAL | 0.50 | 5.01% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1481.00 | 2026-01-19 11:15:00 | 1423.10 | PARTIAL | 0.50 | 3.91% |
| SELL | retest2 | 2026-01-16 10:30:00 | 1492.10 | 2026-01-19 12:15:00 | 1417.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:15:00 | 1498.20 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.45% |
| SELL | retest2 | 2026-01-14 12:30:00 | 1497.90 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1498.00 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.43% |
| SELL | retest2 | 2026-01-16 09:15:00 | 1481.00 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 4.35% |
| SELL | retest2 | 2026-01-16 10:30:00 | 1492.10 | 2026-01-21 12:15:00 | 1416.60 | STOP_HIT | 0.50 | 5.06% |
| BUY | retest1 | 2026-02-05 09:15:00 | 1465.40 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-06 14:00:00 | 1462.60 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-09 09:45:00 | 1468.00 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-02-09 12:00:00 | 1462.30 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-02-09 13:00:00 | 1465.30 | 2026-02-10 09:15:00 | 1447.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2026-02-19 09:45:00 | 1432.90 | 2026-02-19 15:15:00 | 1408.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-02-25 09:15:00 | 1525.00 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-02-25 12:00:00 | 1516.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-26 09:15:00 | 1550.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -3.49% |
| BUY | retest2 | 2026-02-26 13:00:00 | 1518.10 | 2026-03-04 10:15:00 | 1496.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1372.10 | 2026-03-17 14:15:00 | 1392.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-03-17 14:15:00 | 1375.10 | 2026-03-17 14:15:00 | 1392.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-03-25 14:45:00 | 1349.30 | 2026-04-01 13:15:00 | 1338.00 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1345.00 | 2026-04-13 15:15:00 | 1375.10 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2026-05-04 09:15:00 | 1633.10 | 2026-05-04 14:15:00 | 1796.41 | TARGET_HIT | 1.00 | 10.00% |
