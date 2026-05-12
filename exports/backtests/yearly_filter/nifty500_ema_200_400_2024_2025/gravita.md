# Gravita India Ltd. (GRAVITA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1760.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 14
- **Target hits / Stop hits / Partials:** 4 / 16 / 7
- **Avg / median % per leg:** 1.30% / -0.11%
- **Sum % (uncompounded):** 34.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.32% | 26.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 3 | 60.0% | 3 | 2 | 0 | 5.32% | 26.6% |
| SELL (all) | 22 | 10 | 45.5% | 1 | 14 | 7 | 0.38% | 8.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 22 | 10 | 45.5% | 1 | 14 | 7 | 0.38% | 8.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 13 | 48.1% | 4 | 16 | 7 | 1.30% | 35.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 1027.65 | 953.89 | 953.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 11:15:00 | 1092.00 | 962.16 | 958.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 976.25 | 1013.15 | 987.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 976.25 | 1013.15 | 987.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 976.25 | 1013.15 | 987.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 976.25 | 1013.15 | 987.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 932.55 | 1012.35 | 987.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:00:00 | 932.55 | 1012.35 | 987.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1000.00 | 1012.23 | 987.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 1007.95 | 1012.23 | 987.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 15:15:00 | 1005.00 | 1012.32 | 987.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 10:45:00 | 1007.95 | 1011.88 | 987.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-06 15:15:00 | 1108.75 | 1018.16 | 992.59 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2130.00 | 2206.90 | 2207.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 12:15:00 | 2081.90 | 2202.28 | 2204.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 12:15:00 | 1755.00 | 1738.85 | 1861.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 13:00:00 | 1755.00 | 1738.85 | 1861.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1824.00 | 1750.74 | 1847.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 1824.00 | 1750.74 | 1847.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1877.20 | 1752.00 | 1848.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1877.20 | 1752.00 | 1848.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1871.05 | 1753.19 | 1848.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1899.65 | 1753.19 | 1848.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 1863.05 | 1759.06 | 1848.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 1863.05 | 1759.06 | 1848.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1835.00 | 1759.81 | 1848.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:45:00 | 1860.65 | 1759.81 | 1848.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1772.95 | 1731.52 | 1812.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:30:00 | 1820.95 | 1731.52 | 1812.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 1853.00 | 1733.23 | 1812.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 1853.00 | 1733.23 | 1812.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 1884.00 | 1734.73 | 1813.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 1834.40 | 1734.73 | 1813.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 1923.30 | 1736.61 | 1813.65 | SL hit (close>static) qty=1.00 sl=1911.65 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 1976.90 | 1854.90 | 1854.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 2010.00 | 1856.44 | 1855.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1917.10 | 1921.39 | 1893.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 13:30:00 | 1912.70 | 1921.39 | 1893.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 1895.00 | 1920.73 | 1894.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 1890.00 | 1920.73 | 1894.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 1891.40 | 1920.44 | 1894.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:00:00 | 1891.40 | 1920.44 | 1894.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1888.40 | 1920.12 | 1894.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 1891.40 | 1920.12 | 1894.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1859.60 | 1917.54 | 1893.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:30:00 | 1860.70 | 1917.54 | 1893.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 1861.00 | 1907.96 | 1890.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 1861.00 | 1907.96 | 1890.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1878.90 | 1895.77 | 1885.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 1889.60 | 1895.71 | 1885.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 09:30:00 | 1884.70 | 1895.36 | 1885.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 1855.00 | 1894.41 | 1885.51 | SL hit (close<static) qty=1.00 sl=1864.70 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1764.30 | 1877.85 | 1877.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1753.20 | 1876.61 | 1877.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 1803.20 | 1801.32 | 1833.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-26 11:30:00 | 1797.60 | 1801.32 | 1833.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1825.10 | 1801.68 | 1832.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 1821.00 | 1801.68 | 1832.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1851.50 | 1802.17 | 1832.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1851.50 | 1802.17 | 1832.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1862.50 | 1802.77 | 1832.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 1862.50 | 1802.77 | 1832.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 1839.90 | 1821.68 | 1837.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 1839.90 | 1821.68 | 1837.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1825.10 | 1821.72 | 1837.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1830.70 | 1821.72 | 1837.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1808.50 | 1821.58 | 1837.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 1800.00 | 1821.37 | 1837.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 11:30:00 | 1807.00 | 1819.97 | 1835.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-14 09:15:00 | 1716.65 | 1810.34 | 1829.33 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1710.00 | 1789.38 | 1815.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 1809.00 | 1774.14 | 1804.08 | SL hit (close>ema200) qty=0.50 sl=1774.14 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-17 11:15:00)

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

### Cycle 6 — SELL (started 2026-01-19 10:15:00)

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

### Cycle 7 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 1674.00 | 1565.12 | 1564.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1720.70 | 1566.67 | 1565.76 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 12:30:00 | 941.10 | 2024-05-17 09:15:00 | 958.50 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-05-21 10:45:00 | 941.00 | 2024-05-21 12:15:00 | 954.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-21 11:15:00 | 938.95 | 2024-05-21 12:15:00 | 954.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2024-05-22 09:30:00 | 938.20 | 2024-05-22 10:15:00 | 977.00 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1007.95 | 2024-06-06 15:15:00 | 1108.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 15:15:00 | 1005.00 | 2024-06-06 15:15:00 | 1105.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-05 10:45:00 | 1007.95 | 2024-06-06 15:15:00 | 1108.75 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-15 09:15:00 | 1834.40 | 2025-04-15 09:15:00 | 1923.30 | STOP_HIT | 1.00 | -4.85% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1844.30 | 2025-04-30 14:15:00 | 1757.12 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-04-28 15:00:00 | 1849.60 | 2025-04-30 15:15:00 | 1752.08 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-04-29 09:30:00 | 1826.70 | 2025-04-30 15:15:00 | 1735.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:45:00 | 1844.30 | 2025-05-05 09:15:00 | 1961.40 | STOP_HIT | 0.50 | -6.35% |
| SELL | retest2 | 2025-04-28 15:00:00 | 1849.60 | 2025-05-05 09:15:00 | 1961.40 | STOP_HIT | 0.50 | -6.04% |
| SELL | retest2 | 2025-04-29 09:30:00 | 1826.70 | 2025-05-05 09:15:00 | 1961.40 | STOP_HIT | 0.50 | -7.37% |
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
