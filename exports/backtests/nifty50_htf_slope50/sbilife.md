# SBILIFE (SBILIFE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1859.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 6 |
| PENDING | 30 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 3 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 0 / 23 / 0
- **Avg / median % per leg:** -0.84% / -1.26%
- **Sum % (uncompounded):** -19.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 7 | 30.4% | 0 | 23 | 0 | -0.84% | -19.4% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.70% | 5.1% |
| BUY @ 3rd Alert (retest2) | 20 | 4 | 20.0% | 0 | 20 | 0 | -1.22% | -24.5% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.70% | 5.1% |
| retest2 (combined) | 20 | 4 | 20.0% | 0 | 20 | 0 | -1.22% | -24.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1504.35 | 1447.54 | 1447.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1511.45 | 1455.71 | 1451.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 1836.05 | 1838.72 | 1768.58 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 1760.40 | 1830.77 | 1772.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1760.40 | 1830.77 | 1772.98 | EMA400 retest candle locked |
| CROSSOVER_SKIP | 2024-10-28 13:15:00 | 1604.45 | 1743.43 | 1743.79 | HTF filter: close above htf_sma |
| CROSSOVER_SKIP | 2025-03-27 10:15:00 | 1543.25 | 1473.88 | 1473.68 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-05-23 11:15:00 | 1800.00 | 1697.20 | 1628.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 1801.80 | 1698.24 | 1629.85 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-05-26 11:15:00 | 1799.00 | 1704.04 | 1634.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 12:15:00 | 1803.10 | 1705.03 | 1635.65 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1759.20 | 1745.99 | 1677.46 | SL hit qty=1.00 sl=1759.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 1759.20 | 1745.99 | 1677.46 | SL hit qty=1.00 sl=1759.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-11 12:15:00 | 1798.50 | 1753.91 | 1689.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-11 13:15:00 | 1792.00 | 1754.29 | 1689.90 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-11 14:15:00 | 1799.70 | 1754.74 | 1690.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1800.00 | 1755.19 | 1691.00 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1759.20 | 1756.80 | 1694.34 | SL hit qty=1.00 sl=1759.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-06-16 14:15:00 | 1799.10 | 1758.37 | 1698.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-06-16 15:15:00 | 1795.40 | 1758.74 | 1699.25 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-17 09:15:00 | 1798.20 | 1759.13 | 1699.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 10:15:00 | 1798.40 | 1759.52 | 1700.23 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1767.10 | 1811.31 | 1768.77 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1759.20 | 1810.82 | 1768.74 | SL hit qty=1.00 sl=1759.20 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-21 14:15:00 | 1804.40 | 1808.54 | 1769.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 1801.50 | 1808.47 | 1769.96 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-22 10:15:00 | 1800.10 | 1808.26 | 1770.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 1807.10 | 1808.25 | 1770.43 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-23 10:15:00 | 1807.00 | 1808.12 | 1771.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:15:00 | 1810.10 | 1808.14 | 1771.67 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-24 15:15:00 | 1819.90 | 1807.34 | 1773.22 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1837.10 | 1807.64 | 1773.54 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 1080m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1790.30 | 1815.96 | 1784.69 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-08-04 10:15:00 | 1813.60 | 1815.94 | 1784.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 1814.00 | 1815.92 | 1784.98 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1784.30 | 1831.04 | 1812.84 | SL hit qty=1.00 sl=1784.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-05 09:15:00 | 1806.60 | 1829.20 | 1812.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 1816.10 | 1829.07 | 1812.38 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1784.30 | 1827.58 | 1812.11 | SL hit qty=1.00 sl=1784.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-09 14:15:00 | 1805.00 | 1823.17 | 1810.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 1805.70 | 1822.99 | 1810.71 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1784.30 | 1820.75 | 1814.50 | SL hit qty=1.00 sl=1784.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1764.40 | 1813.73 | 1811.44 | SL hit qty=1.00 sl=1764.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1764.40 | 1813.73 | 1811.44 | SL hit qty=1.00 sl=1764.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1764.40 | 1813.73 | 1811.44 | SL hit qty=1.00 sl=1764.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1764.40 | 1813.73 | 1811.44 | SL hit qty=1.00 sl=1764.40 alert=retest2 |
| CROSSOVER_SKIP | 2025-10-07 15:15:00 | 1784.10 | 1809.06 | 1809.16 | HTF filter: close above htf_sma |
| Cross detected — sustain check pending | 2025-10-09 13:15:00 | 1808.40 | 1805.63 | 1807.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 1810.70 | 1805.68 | 1807.40 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1813.00 | 1805.76 | 1807.43 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-10 09:15:00 | 1828.00 | 1805.98 | 1807.53 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1837.00 | 1806.29 | 1807.68 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-10 15:15:00 | 1806.40 | 1806.87 | 1807.94 | SL hit qty=1.00 sl=1806.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-13 09:15:00 | 1821.20 | 1807.01 | 1808.01 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 1816.20 | 1807.10 | 1808.05 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-13 13:15:00 | 1814.40 | 1807.28 | 1808.12 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 14:15:00 | 1815.60 | 1807.36 | 1808.16 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 14:15:00 | 1816.10 | 1807.77 | 1808.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 1815.00 | 1807.84 | 1808.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.14 | 1815.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.26 | 1915.85 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2025-12-01 15:15:00 | 1974.00 | 1964.25 | 1917.74 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 09:15:00 | 1960.80 | 1964.22 | 1917.95 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-02 13:15:00 | 1975.40 | 1964.32 | 1918.92 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1980.30 | 1964.48 | 1919.23 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 1976.80 | 1965.01 | 1920.39 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 12:15:00 | 1974.70 | 1965.10 | 1920.66 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 14:15:00 | 1972.90 | 1965.20 | 1921.15 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 15:15:00 | 1974.00 | 1965.29 | 1921.41 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.38 | 2009.85 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 2009.85 | 2046.38 | 2009.85 | SL hit qty=1.00 sl=2009.85 alert=retest1 |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 2030.30 | 2043.03 | 2009.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 2025.20 | 2042.85 | 2009.82 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 2028.90 | 2042.71 | 2009.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 2034.00 | 2042.63 | 2010.04 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 2009.70 | 2043.00 | 2011.97 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2051.70 | 2033.71 | 2010.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2050.30 | 2033.87 | 2010.35 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 13:15:00 | 2009.70 | 2033.71 | 2010.62 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 12:15:00 | 2041.40 | 2032.85 | 2010.86 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2042.50 | 2032.95 | 2011.02 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-05 12:15:00 | 2009.70 | 2032.55 | 2011.47 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-09 10:15:00 | 2033.40 | 2029.25 | 2011.00 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 11:15:00 | 2025.40 | 2029.21 | 2011.07 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-11 13:15:00 | 2028.60 | 2027.62 | 2011.63 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-11 14:15:00 | 2026.40 | 2027.61 | 2011.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2031.30 | 2026.79 | 2011.98 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-13 10:15:00 | 2020.70 | 2026.73 | 2012.03 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 13:15:00 | 2031.90 | 2026.65 | 2012.21 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2032.60 | 2026.71 | 2012.31 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2011.90 | 2047.42 | 2028.63 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2009.70 | 2047.42 | 2028.63 | SL hit qty=1.00 sl=2009.70 alert=retest2 |
| CROSSOVER_SKIP | 2026-03-10 15:15:00 | 1963.70 | 2013.66 | 2013.68 | HTF filter: close above htf_sma |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-23 12:15:00 | 1801.80 | 2025-06-06 09:15:00 | 1759.20 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-26 12:15:00 | 1803.10 | 2025-06-06 09:15:00 | 1759.20 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-06-11 15:15:00 | 1800.00 | 2025-06-13 09:15:00 | 1759.20 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-06-17 10:15:00 | 1798.40 | 2025-07-18 10:15:00 | 1759.20 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-07-21 15:15:00 | 1801.50 | 2025-09-04 11:15:00 | 1784.30 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-07-22 11:15:00 | 1807.10 | 2025-09-08 09:15:00 | 1784.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-23 11:15:00 | 1810.10 | 2025-09-30 09:15:00 | 1784.30 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-07-25 09:15:00 | 1837.10 | 2025-10-06 09:15:00 | 1764.40 | STOP_HIT | 1.00 | -3.96% |
| BUY | retest2 | 2025-08-04 11:15:00 | 1814.00 | 2025-10-06 09:15:00 | 1764.40 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-09-05 10:15:00 | 1816.10 | 2025-10-06 09:15:00 | 1764.40 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-09-09 15:15:00 | 1805.70 | 2025-10-06 09:15:00 | 1764.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-10-09 14:15:00 | 1810.70 | 2025-10-10 15:15:00 | 1806.40 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-10-10 10:15:00 | 1837.00 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2025-10-13 10:15:00 | 1816.20 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-10-13 14:15:00 | 1815.60 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 1.00 | 1.89% |
| BUY | retest2 | 2025-10-14 15:15:00 | 1815.00 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 1.00 | 1.93% |
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.70 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest1 | 2025-12-03 15:15:00 | 1974.00 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2034.00 | 2026-01-29 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2050.30 | 2026-02-03 13:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2042.50 | 2026-02-05 12:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2032.60 | 2026-03-02 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.13% |
