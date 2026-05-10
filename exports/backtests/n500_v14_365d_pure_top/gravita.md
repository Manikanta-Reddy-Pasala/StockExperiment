# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1760.60
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
| ALERT2_SKIP | 0 |
| ALERT3 | 28 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 13 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 6
- **Target hits / Stop hits / Partials:** 1 / 8 / 4
- **Avg / median % per leg:** 1.81% / 1.98%
- **Sum % (uncompounded):** 23.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| SELL (all) | 11 | 7 | 63.6% | 1 | 6 | 4 | 2.45% | 27.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 11 | 7 | 63.6% | 1 | 6 | 4 | 2.45% | 27.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 13 | 7 | 53.8% | 1 | 8 | 4 | 1.81% | 23.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 1976.90 | 1854.90 | 1854.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 2010.00 | 1856.44 | 1855.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1917.10 | 1921.39 | 1893.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:30:00 | 1912.70 | 1921.39 | 1893.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1895.00 | 1920.73 | 1894.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 1890.00 | 1920.73 | 1894.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1891.40 | 1920.44 | 1894.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1891.40 | 1920.44 | 1894.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1888.40 | 1920.12 | 1894.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 1891.40 | 1920.12 | 1894.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1859.60 | 1917.54 | 1893.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1860.70 | 1917.54 | 1893.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1861.00 | 1907.96 | 1890.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 1861.00 | 1907.96 | 1890.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1878.90 | 1895.77 | 1885.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 1889.60 | 1895.71 | 1885.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1884.70 | 1895.36 | 1885.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 1855.00 | 1894.41 | 1885.49 | SL hit (close<static) qty=1.00 sl=1864.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 1855.00 | 1894.41 | 1885.49 | SL hit (close<static) qty=1.00 sl=1864.70 alert=retest2 |

### Cycle 2 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1764.30 | 1877.85 | 1877.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1753.20 | 1876.61 | 1877.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 1803.20 | 1801.32 | 1833.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:30:00 | 1797.60 | 1801.32 | 1833.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1825.10 | 1801.68 | 1832.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1821.00 | 1801.68 | 1832.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1851.50 | 1802.17 | 1832.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1851.50 | 1802.17 | 1832.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1862.50 | 1802.77 | 1832.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 1862.50 | 1802.77 | 1832.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1839.90 | 1821.68 | 1837.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1839.90 | 1821.68 | 1837.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1825.10 | 1821.72 | 1837.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1830.70 | 1821.72 | 1837.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1808.50 | 1821.58 | 1837.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 1800.00 | 1821.37 | 1837.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 1807.00 | 1819.97 | 1835.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 1716.65 | 1810.34 | 1829.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1710.00 | 1789.38 | 1815.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1809.00 | 1774.14 | 1804.07 | SL hit (close>ema200) qty=0.50 sl=1774.14 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1809.00 | 1774.14 | 1804.07 | SL hit (close>ema200) qty=0.50 sl=1774.14 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1806.00 | 1774.14 | 1804.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 1806.00 | 1774.14 | 1804.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1808.20 | 1774.48 | 1804.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 1804.00 | 1774.48 | 1804.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1804.00 | 1774.77 | 1804.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 1813.20 | 1774.77 | 1804.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 1810.20 | 1775.12 | 1804.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:45:00 | 1814.00 | 1775.12 | 1804.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 1809.10 | 1775.46 | 1804.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 1809.50 | 1775.46 | 1804.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 1818.00 | 1776.25 | 1804.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 1844.70 | 1776.25 | 1804.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 1867.00 | 1780.26 | 1805.60 | SL hit (close>static) qty=1.00 sl=1859.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-25 13:15:00 | 1867.00 | 1780.26 | 1805.60 | SL hit (close>static) qty=1.00 sl=1859.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 1823.70 | 1784.96 | 1806.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:15:00 | 1832.00 | 1784.96 | 1806.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1807.40 | 1803.14 | 1813.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1804.60 | 1803.14 | 1813.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1805.00 | 1803.16 | 1813.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 10:30:00 | 1811.80 | 1803.16 | 1813.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1798.00 | 1799.35 | 1810.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 1798.00 | 1799.35 | 1810.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1837.50 | 1799.73 | 1810.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 1837.50 | 1799.73 | 1810.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1849.00 | 1800.22 | 1810.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:30:00 | 1819.30 | 1800.30 | 1810.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 13:15:00 | 1728.33 | 1794.81 | 1806.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-28 09:15:00 | 1637.37 | 1770.11 | 1790.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1754.50 | 1683.94 | 1683.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 1783.90 | 1694.75 | 1689.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1815.00 | 1817.84 | 1780.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:15:00 | 1810.00 | 1817.84 | 1780.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1787.00 | 1816.71 | 1782.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1787.00 | 1816.71 | 1782.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1782.00 | 1816.36 | 1782.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1774.90 | 1816.36 | 1782.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1755.20 | 1815.75 | 1782.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1755.20 | 1815.75 | 1782.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1561.50 | 1754.53 | 1755.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1547.80 | 1737.27 | 1746.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 1674.90 | 1651.42 | 1693.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 11:00:00 | 1674.90 | 1651.42 | 1693.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1719.20 | 1653.66 | 1687.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1712.00 | 1654.26 | 1687.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1626.40 | 1659.17 | 1686.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 14:15:00 | 1671.60 | 1658.72 | 1685.47 | SL hit (close>ema200) qty=0.50 sl=1658.72 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 1707.80 | 1558.47 | 1561.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1720.70 | 1566.67 | 1565.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-06 15:00:00 | 1889.60 | 2025-06-09 12:15:00 | 1855.00 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-06-09 09:30:00 | 1884.70 | 2025-06-09 12:15:00 | 1855.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1800.00 | 2025-07-14 09:15:00 | 1716.65 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2025-07-09 11:30:00 | 1807.00 | 2025-07-18 10:15:00 | 1710.00 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1800.00 | 2025-07-24 09:15:00 | 1809.00 | STOP_HIT | 0.50 | -0.50% |
| SELL | retest2 | 2025-07-09 11:30:00 | 1807.00 | 2025-07-24 09:15:00 | 1809.00 | STOP_HIT | 0.50 | -0.11% |
| SELL | retest2 | 2025-07-24 09:45:00 | 1806.00 | 2025-07-25 13:15:00 | 1867.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-07-24 10:15:00 | 1806.00 | 2025-07-25 13:15:00 | 1867.00 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-08-08 09:30:00 | 1819.30 | 2025-08-18 13:15:00 | 1728.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 09:30:00 | 1819.30 | 2025-08-28 09:15:00 | 1637.37 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-10 10:45:00 | 1712.00 | 2026-02-16 09:15:00 | 1626.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 10:45:00 | 1712.00 | 2026-02-16 14:15:00 | 1671.60 | STOP_HIT | 0.50 | 2.36% |
| SELL | retest2 | 2026-05-04 10:00:00 | 1707.80 | 2026-05-04 15:15:00 | 1674.00 | STOP_HIT | 1.00 | 1.98% |
