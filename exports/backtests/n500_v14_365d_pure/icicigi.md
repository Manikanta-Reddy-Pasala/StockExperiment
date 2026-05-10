# ICICI Lombard General Insurance Company Ltd. (ICICIGI)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1820.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 32 |
| PARTIAL | 8 |
| TARGET_HIT | 10 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 40 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 20
- **Target hits / Stop hits / Partials:** 10 / 22 / 8
- **Avg / median % per leg:** 2.79% / 4.13%
- **Sum % (uncompounded):** 111.49%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 12 | 4 | 33.3% | 4 | 8 | 0 | 2.64% | 31.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 4 | 8 | 0 | 2.64% | 31.7% |
| SELL (all) | 28 | 16 | 57.1% | 6 | 14 | 8 | 2.85% | 79.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 16 | 57.1% | 6 | 14 | 8 | 2.85% | 79.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 40 | 20 | 50.0% | 10 | 22 | 8 | 2.79% | 111.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 10:15:00 | 1847.80 | 1923.02 | 1923.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-01 10:15:00 | 1832.50 | 1917.64 | 1920.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 1898.10 | 1883.97 | 1899.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 13:00:00 | 1898.10 | 1883.97 | 1899.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1893.00 | 1884.18 | 1899.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 1898.60 | 1884.18 | 1899.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 1900.10 | 1884.34 | 1899.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 09:45:00 | 1889.20 | 1884.35 | 1899.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1908.40 | 1885.76 | 1899.81 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:45:00 | 1885.60 | 1886.88 | 1899.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 11:15:00 | 1889.20 | 1884.52 | 1896.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 1889.10 | 1884.62 | 1896.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 1895.00 | 1884.73 | 1896.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 1895.00 | 1884.73 | 1896.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 1895.00 | 1884.83 | 1896.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 1900.00 | 1884.83 | 1896.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1884.80 | 1884.93 | 1896.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 09:15:00 | 1861.00 | 1884.43 | 1896.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 1900.10 | 1884.17 | 1895.62 | SL hit (close>static) qty=1.00 sl=1899.30 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 1866.20 | 1885.65 | 1895.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 1900.00 | 1885.68 | 1895.17 | SL hit (close>static) qty=1.00 sl=1899.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1908.00 | 1886.46 | 1895.33 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1908.00 | 1886.46 | 1895.33 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 1908.00 | 1886.46 | 1895.33 | SL hit (close>static) qty=1.00 sl=1907.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 1860.30 | 1891.75 | 1897.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:00:00 | 1870.00 | 1890.45 | 1896.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2000.00 | 1884.91 | 1892.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2000.00 | 1884.91 | 1892.71 | SL hit (close>static) qty=1.00 sl=1899.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 2000.00 | 1884.91 | 1892.71 | SL hit (close>static) qty=1.00 sl=1899.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 2000.00 | 1884.91 | 1892.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 15:15:00 | 2007.20 | 1900.19 | 1900.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 2027.80 | 1908.38 | 1904.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 1995.70 | 1996.81 | 1966.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 1995.70 | 1996.81 | 1966.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1970.70 | 1996.28 | 1969.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:45:00 | 1981.60 | 1993.76 | 1969.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1956.40 | 1993.32 | 1969.96 | SL hit (close<static) qty=1.00 sl=1965.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:30:00 | 1982.00 | 1992.77 | 1970.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 1955.50 | 1992.23 | 1970.21 | SL hit (close<static) qty=1.00 sl=1965.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 1981.40 | 1990.72 | 1970.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 13:00:00 | 1981.50 | 1990.48 | 1970.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1971.00 | 1989.94 | 1970.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1971.00 | 1989.94 | 1970.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1978.70 | 1989.82 | 1970.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 1961.50 | 1989.08 | 1970.74 | SL hit (close<static) qty=1.00 sl=1965.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 1961.50 | 1989.08 | 1970.74 | SL hit (close<static) qty=1.00 sl=1965.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1950.00 | 1960.04 | 1960.07 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1979.90 | 1960.23 | 1960.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1988.40 | 1960.51 | 1960.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1946.50 | 1965.51 | 1962.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 1946.50 | 1965.51 | 1962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1921.30 | 1965.07 | 1962.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1921.30 | 1965.07 | 1962.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1895.40 | 1960.57 | 1960.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-13 12:15:00 | 1874.00 | 1951.38 | 1955.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 13:15:00 | 1875.50 | 1873.77 | 1904.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 13:30:00 | 1876.20 | 1873.77 | 1904.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1899.90 | 1873.24 | 1901.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 1899.90 | 1873.24 | 1901.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 1899.00 | 1873.50 | 1901.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 1902.50 | 1873.50 | 1901.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 1905.90 | 1873.82 | 1901.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:00:00 | 1905.90 | 1873.82 | 1901.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1911.30 | 1874.20 | 1901.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:45:00 | 1910.00 | 1874.20 | 1901.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1895.40 | 1875.04 | 1901.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:15:00 | 1892.80 | 1907.44 | 1911.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 13:45:00 | 1890.70 | 1906.96 | 1911.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:15:00 | 1890.90 | 1906.96 | 1911.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 14:45:00 | 1892.70 | 1906.83 | 1911.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1881.30 | 1896.84 | 1905.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 09:15:00 | 1876.50 | 1896.60 | 1905.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 15:00:00 | 1875.00 | 1895.84 | 1904.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1798.16 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1796.36 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 10:15:00 | 1798.07 | 1880.83 | 1895.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:15:00 | 1796.16 | 1877.45 | 1892.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 1782.67 | 1868.54 | 1887.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 14:15:00 | 1781.25 | 1868.54 | 1887.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-24 10:15:00 | 1703.52 | 1855.13 | 1879.71 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-24 10:15:00 | 1703.43 | 1855.13 | 1879.71 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 1701.63 | 1835.83 | 1867.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 1701.81 | 1835.83 | 1867.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 1688.85 | 1835.83 | 1867.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-30 09:15:00 | 1687.50 | 1835.83 | 1867.24 | Target hit (10%) qty=0.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 10:30:00 | 1878.50 | 1796.51 | 1833.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 1875.00 | 1800.97 | 1834.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 1825.00 | 1813.74 | 1837.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1824.00 | 1813.74 | 1837.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:00:00 | 1822.90 | 1813.96 | 1837.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1811.70 | 1814.48 | 1837.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 13:30:00 | 1818.50 | 1814.77 | 1837.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 1784.57 | 1813.96 | 1836.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 1781.25 | 1813.96 | 1836.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1797.50 | 1796.93 | 1821.91 | SL hit (close>ema200) qty=0.50 sl=1796.93 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 1797.50 | 1796.93 | 1821.91 | SL hit (close>ema200) qty=0.50 sl=1796.93 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1821.10 | 1797.22 | 1821.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1821.10 | 1797.22 | 1821.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1830.10 | 1797.93 | 1821.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 1830.10 | 1797.93 | 1821.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1836.00 | 1798.31 | 1821.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1836.00 | 1798.31 | 1821.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1846.70 | 1798.79 | 1821.87 | SL hit (close>static) qty=1.00 sl=1840.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1846.70 | 1798.79 | 1821.87 | SL hit (close>static) qty=1.00 sl=1840.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1846.70 | 1798.79 | 1821.87 | SL hit (close>static) qty=1.00 sl=1840.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 11:15:00 | 1846.70 | 1798.79 | 1821.87 | SL hit (close>static) qty=1.00 sl=1840.10 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1822.10 | 1802.45 | 1822.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1822.10 | 1802.45 | 1822.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1826.40 | 1802.69 | 1822.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 1826.40 | 1802.69 | 1822.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1820.00 | 1802.86 | 1822.74 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-28 13:30:00 | 1858.40 | 2025-06-30 10:15:00 | 2040.94 | TARGET_HIT | 1.00 | 9.82% |
| BUY | retest2 | 2025-05-28 14:00:00 | 1859.50 | 2025-06-30 11:15:00 | 2044.24 | TARGET_HIT | 1.00 | 9.93% |
| BUY | retest2 | 2025-05-29 11:45:00 | 1855.40 | 2025-06-30 11:15:00 | 2045.45 | TARGET_HIT | 1.00 | 10.24% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1859.30 | 2025-06-30 11:15:00 | 2045.23 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1966.10 | 2025-08-19 09:15:00 | 1928.20 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-19 12:00:00 | 1950.10 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-08-19 14:00:00 | 1946.20 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-19 14:30:00 | 1949.90 | 2025-08-22 11:15:00 | 1937.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-15 09:45:00 | 1889.20 | 2025-09-16 11:15:00 | 1908.40 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-09-17 12:45:00 | 1885.60 | 2025-09-26 09:15:00 | 1900.10 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-23 11:15:00 | 1889.20 | 2025-10-01 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1889.10 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-25 09:15:00 | 1861.00 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-10-01 09:45:00 | 1866.20 | 2025-10-03 09:15:00 | 1908.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-10-09 09:15:00 | 1860.30 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -7.51% |
| SELL | retest2 | 2025-10-10 13:00:00 | 1870.00 | 2025-10-15 09:15:00 | 2000.00 | STOP_HIT | 1.00 | -6.95% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1981.60 | 2025-12-02 09:15:00 | 1956.40 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-12-02 12:30:00 | 1982.00 | 2025-12-03 09:15:00 | 1955.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-04 10:30:00 | 1981.40 | 2025-12-08 09:15:00 | 1961.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-04 13:00:00 | 1981.50 | 2025-12-08 09:15:00 | 1961.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1892.80 | 2026-03-18 10:15:00 | 1798.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 13:45:00 | 1890.70 | 2026-03-18 10:15:00 | 1796.36 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2026-03-02 14:15:00 | 1890.90 | 2026-03-18 10:15:00 | 1798.07 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-03-02 14:45:00 | 1892.70 | 2026-03-19 09:15:00 | 1796.16 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1876.50 | 2026-03-20 14:15:00 | 1782.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 1875.00 | 2026-03-20 14:15:00 | 1781.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 15:15:00 | 1892.80 | 2026-03-24 10:15:00 | 1703.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 13:45:00 | 1890.70 | 2026-03-24 10:15:00 | 1703.43 | TARGET_HIT | 0.50 | 9.90% |
| SELL | retest2 | 2026-03-02 14:15:00 | 1890.90 | 2026-03-30 09:15:00 | 1701.63 | TARGET_HIT | 0.50 | 10.01% |
| SELL | retest2 | 2026-03-02 14:45:00 | 1892.70 | 2026-03-30 09:15:00 | 1701.81 | TARGET_HIT | 0.50 | 10.09% |
| SELL | retest2 | 2026-03-11 09:15:00 | 1876.50 | 2026-03-30 09:15:00 | 1688.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-11 15:00:00 | 1875.00 | 2026-03-30 09:15:00 | 1687.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 10:30:00 | 1878.50 | 2026-04-24 11:15:00 | 1784.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 09:15:00 | 1875.00 | 2026-04-24 11:15:00 | 1781.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-16 10:30:00 | 1878.50 | 2026-05-06 10:15:00 | 1797.50 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2026-04-17 09:15:00 | 1875.00 | 2026-05-06 10:15:00 | 1797.50 | STOP_HIT | 0.50 | 4.13% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1824.00 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2026-04-22 12:00:00 | 1822.90 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1811.70 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-04-23 13:30:00 | 1818.50 | 2026-05-07 11:15:00 | 1846.70 | STOP_HIT | 1.00 | -1.55% |
