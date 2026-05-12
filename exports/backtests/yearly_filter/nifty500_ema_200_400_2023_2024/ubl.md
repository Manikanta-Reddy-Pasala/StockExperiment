# United Breweries Ltd. (UBL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1419.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 62 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 66 |
| PARTIAL | 1 |
| TARGET_HIT | 13 |
| STOP_HIT | 53 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 67 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 50
- **Target hits / Stop hits / Partials:** 13 / 53 / 1
- **Avg / median % per leg:** 0.55% / -1.28%
- **Sum % (uncompounded):** 37.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 15 | 40.5% | 12 | 25 | 0 | 1.94% | 71.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 15 | 40.5% | 12 | 25 | 0 | 1.94% | 71.6% |
| SELL (all) | 30 | 2 | 6.7% | 1 | 28 | 1 | -1.15% | -34.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 30 | 2 | 6.7% | 1 | 28 | 1 | -1.15% | -34.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 67 | 17 | 25.4% | 13 | 53 | 1 | 0.55% | 37.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 13:15:00 | 1706.35 | 1738.64 | 1738.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 15:15:00 | 1699.70 | 1737.88 | 1738.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 1733.50 | 1730.46 | 1734.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-12 10:00:00 | 1733.50 | 1730.46 | 1734.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 10:15:00 | 1709.35 | 1730.25 | 1734.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 10:30:00 | 1739.75 | 1730.25 | 1734.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 12:15:00 | 1741.00 | 1730.22 | 1734.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-12 13:00:00 | 1741.00 | 1730.22 | 1734.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 1728.00 | 1730.20 | 1734.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-12 14:15:00 | 1723.60 | 1730.20 | 1734.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 10:15:00 | 1724.80 | 1729.92 | 1733.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-13 10:45:00 | 1722.95 | 1729.82 | 1733.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-14 10:45:00 | 1720.50 | 1728.32 | 1732.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 13:15:00 | 1724.50 | 1726.62 | 1731.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:00:00 | 1724.50 | 1726.62 | 1731.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1733.60 | 1726.69 | 1731.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-15 15:00:00 | 1733.60 | 1726.69 | 1731.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 15:15:00 | 1741.05 | 1726.84 | 1731.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:15:00 | 1724.05 | 1726.84 | 1731.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 1730.70 | 1726.42 | 1731.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:00:00 | 1730.70 | 1726.42 | 1731.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 13:15:00 | 1734.70 | 1726.50 | 1731.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:30:00 | 1733.60 | 1726.50 | 1731.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 14:15:00 | 1729.60 | 1726.53 | 1731.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 15:15:00 | 1728.00 | 1726.53 | 1731.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 15:15:00 | 1728.00 | 1726.55 | 1731.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 09:15:00 | 1717.45 | 1726.55 | 1731.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 12:00:00 | 1725.20 | 1721.62 | 1728.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-22 15:15:00 | 1718.20 | 1721.95 | 1728.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 09:30:00 | 1722.40 | 1721.91 | 1728.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 11:15:00 | 1728.00 | 1722.01 | 1728.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-26 11:30:00 | 1729.10 | 1722.01 | 1728.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 12:15:00 | 1726.00 | 1722.05 | 1728.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-26 13:30:00 | 1716.95 | 1722.03 | 1728.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-28 14:15:00 | 1739.00 | 1720.04 | 1726.79 | SL hit (close>static) qty=1.00 sl=1736.30 alert=retest2 |

### Cycle 2 — BUY (started 2024-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 13:15:00 | 1811.90 | 1732.95 | 1732.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 11:15:00 | 1842.10 | 1744.74 | 1739.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-13 09:15:00 | 1903.75 | 1908.00 | 1847.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-13 09:45:00 | 1891.90 | 1908.00 | 1847.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1854.10 | 1905.20 | 1859.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:00:00 | 1854.10 | 1905.20 | 1859.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1850.40 | 1904.66 | 1859.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 1850.60 | 1904.66 | 1859.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1861.05 | 1902.13 | 1860.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 1861.05 | 1902.13 | 1860.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1863.35 | 1898.60 | 1860.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:30:00 | 1864.95 | 1898.60 | 1860.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1874.95 | 1898.09 | 1860.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 10:15:00 | 1885.85 | 1887.73 | 1861.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 1850.65 | 1887.36 | 1861.25 | SL hit (close<static) qty=1.00 sl=1859.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 1953.70 | 2043.86 | 2044.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-28 09:15:00 | 1940.60 | 2040.12 | 2042.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1942.90 | 1932.06 | 1970.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:30:00 | 1935.35 | 1932.06 | 1970.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1956.85 | 1936.18 | 1967.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:15:00 | 1954.00 | 1936.18 | 1967.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 15:15:00 | 1950.25 | 1936.70 | 1967.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 09:30:00 | 1952.50 | 1936.97 | 1966.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 09:15:00 | 1973.90 | 1938.71 | 1966.80 | SL hit (close>static) qty=1.00 sl=1973.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 2039.70 | 1981.64 | 1981.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 09:15:00 | 2055.95 | 1992.48 | 1987.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 13:15:00 | 1960.95 | 2024.23 | 2005.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 1960.95 | 2024.23 | 2005.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 1960.95 | 2024.23 | 2005.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 1960.95 | 2024.23 | 2005.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 2010.85 | 2024.10 | 2005.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 09:15:00 | 2013.30 | 2023.77 | 2005.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 10:15:00 | 2022.00 | 2023.65 | 2005.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 11:15:00 | 2015.35 | 2023.53 | 2005.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 12:30:00 | 2014.15 | 2023.27 | 2005.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 2015.00 | 2023.18 | 2005.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 15:00:00 | 2028.00 | 2023.23 | 2005.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 15:15:00 | 2002.00 | 2023.02 | 2005.92 | SL hit (close<static) qty=1.00 sl=2004.95 alert=retest2 |

### Cycle 5 — SELL (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 11:15:00 | 1915.85 | 2027.37 | 2027.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-05 09:15:00 | 1891.30 | 2021.57 | 2024.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 09:15:00 | 1992.75 | 1959.00 | 1986.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 1992.75 | 1959.00 | 1986.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1992.75 | 1959.00 | 1986.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 1992.75 | 1959.00 | 1986.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 1975.05 | 1959.16 | 1986.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 11:30:00 | 1968.40 | 1959.29 | 1986.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 12:15:00 | 1968.10 | 1959.29 | 1986.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 11:15:00 | 1969.80 | 1959.14 | 1985.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 12:00:00 | 1965.80 | 1959.21 | 1985.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1976.20 | 1955.07 | 1980.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 1976.20 | 1955.07 | 1980.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1975.00 | 1955.26 | 1980.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 1981.50 | 1955.26 | 1980.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1984.10 | 1955.75 | 1980.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2000.80 | 1955.75 | 1980.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2002.65 | 1956.21 | 1980.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 2003.70 | 1956.21 | 1980.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-03-28 10:15:00 | 2013.25 | 1956.78 | 1980.28 | SL hit (close>static) qty=1.00 sl=2011.15 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 2130.00 | 1991.94 | 1991.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 2145.90 | 1993.47 | 1992.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 2096.20 | 2103.80 | 2063.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 11:00:00 | 2096.20 | 2103.80 | 2063.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 2052.00 | 2103.29 | 2063.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:00:00 | 2052.00 | 2103.29 | 2063.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 2045.00 | 2102.71 | 2063.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 15:15:00 | 2070.10 | 2101.37 | 2063.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:45:00 | 2054.50 | 2096.35 | 2063.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:00:00 | 2053.80 | 2095.46 | 2063.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:15:00 | 2060.10 | 2093.03 | 2063.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 2053.00 | 2092.23 | 2063.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:30:00 | 2051.10 | 2092.23 | 2063.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 2046.30 | 2086.55 | 2062.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 2040.20 | 2086.55 | 2062.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 2017.90 | 2083.15 | 2061.02 | SL hit (close<static) qty=1.00 sl=2018.30 alert=retest2 |

### Cycle 7 — SELL (started 2025-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 12:15:00 | 2000.00 | 2047.83 | 2047.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1990.20 | 2047.26 | 2047.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1985.20 | 1984.67 | 2005.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:00:00 | 1985.20 | 1984.67 | 2005.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1992.90 | 1984.13 | 2004.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1992.90 | 1984.13 | 2004.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1996.30 | 1984.41 | 2004.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:00:00 | 1994.30 | 1989.10 | 2005.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 2009.50 | 1989.44 | 2005.61 | SL hit (close>static) qty=1.00 sl=2006.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 1694.20 | 1603.59 | 1603.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 09:15:00 | 1551.80 | 1604.61 | 1604.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1541.50 | 1601.63 | 1603.22 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-03-12 14:15:00 | 1723.60 | 2024-03-28 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-03-13 10:15:00 | 1724.80 | 2024-03-28 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-03-13 10:45:00 | 1722.95 | 2024-03-28 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-03-14 10:45:00 | 1720.50 | 2024-03-28 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2024-03-19 09:15:00 | 1717.45 | 2024-03-28 14:15:00 | 1739.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2024-03-22 12:00:00 | 1725.20 | 2024-04-01 10:15:00 | 1760.25 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2024-03-22 15:15:00 | 1718.20 | 2024-04-01 10:15:00 | 1760.25 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2024-03-26 09:30:00 | 1722.40 | 2024-04-01 10:15:00 | 1760.25 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-03-26 13:30:00 | 1716.95 | 2024-04-01 10:15:00 | 1760.25 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2024-06-04 10:15:00 | 1885.85 | 2024-06-04 10:15:00 | 1850.65 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-06-04 13:15:00 | 1908.00 | 2024-06-07 11:15:00 | 2098.80 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-04 10:15:00 | 1954.00 | 2024-12-06 09:15:00 | 1973.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-12-04 15:15:00 | 1950.25 | 2024-12-06 09:15:00 | 1973.90 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-12-05 09:30:00 | 1952.50 | 2024-12-06 09:15:00 | 1973.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-12-09 11:15:00 | 1953.05 | 2024-12-10 09:15:00 | 1990.90 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-01-09 09:15:00 | 2013.30 | 2025-01-09 15:15:00 | 2002.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-01-09 10:15:00 | 2022.00 | 2025-01-13 09:15:00 | 1998.95 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-01-09 11:15:00 | 2015.35 | 2025-02-01 12:15:00 | 2214.63 | TARGET_HIT | 1.00 | 9.89% |
| BUY | retest2 | 2025-01-09 12:30:00 | 2014.15 | 2025-02-01 12:15:00 | 2224.20 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2025-01-09 15:00:00 | 2028.00 | 2025-02-01 12:15:00 | 2216.89 | TARGET_HIT | 1.00 | 9.31% |
| BUY | retest2 | 2025-01-10 14:30:00 | 2021.75 | 2025-02-01 12:15:00 | 2215.57 | TARGET_HIT | 1.00 | 9.59% |
| BUY | retest2 | 2025-01-20 11:00:00 | 2069.70 | 2025-02-01 12:15:00 | 2224.86 | TARGET_HIT | 1.00 | 7.50% |
| BUY | retest2 | 2025-01-20 11:30:00 | 2042.05 | 2025-02-01 13:15:00 | 2246.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-22 10:15:00 | 2074.00 | 2025-02-03 09:15:00 | 2276.67 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2025-01-23 10:00:00 | 2078.80 | 2025-02-03 09:15:00 | 2281.40 | TARGET_HIT | 1.00 | 9.75% |
| BUY | retest2 | 2025-01-23 12:00:00 | 2070.70 | 2025-02-03 09:15:00 | 2286.68 | TARGET_HIT | 1.00 | 10.43% |
| BUY | retest2 | 2025-01-23 13:30:00 | 2069.00 | 2025-02-03 09:15:00 | 2277.77 | TARGET_HIT | 1.00 | 10.09% |
| BUY | retest2 | 2025-01-28 11:30:00 | 2022.60 | 2025-02-03 09:15:00 | 2275.90 | TARGET_HIT | 1.00 | 12.52% |
| BUY | retest2 | 2025-02-14 09:30:00 | 2059.20 | 2025-02-25 10:15:00 | 2031.35 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-02-18 09:15:00 | 2032.05 | 2025-02-27 12:15:00 | 2035.75 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-02-18 12:15:00 | 2024.35 | 2025-02-27 12:15:00 | 2035.75 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-02-20 10:30:00 | 2033.00 | 2025-02-27 12:15:00 | 2035.75 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2025-02-20 13:30:00 | 2033.35 | 2025-02-27 13:15:00 | 2029.85 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-02-20 14:00:00 | 2034.95 | 2025-02-27 13:15:00 | 2029.85 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-02-21 09:15:00 | 2039.10 | 2025-02-27 13:15:00 | 2029.85 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-02-25 09:30:00 | 2047.85 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2025-02-25 11:15:00 | 2052.10 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest2 | 2025-02-25 12:00:00 | 2051.20 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -4.20% |
| BUY | retest2 | 2025-02-25 12:30:00 | 2050.00 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2025-02-27 09:15:00 | 2062.15 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2025-02-27 10:00:00 | 2042.00 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -3.77% |
| BUY | retest2 | 2025-02-27 11:00:00 | 2043.70 | 2025-02-27 15:15:00 | 1965.00 | STOP_HIT | 1.00 | -3.85% |
| SELL | retest2 | 2025-03-20 11:30:00 | 1968.40 | 2025-03-28 10:15:00 | 2013.25 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-03-20 12:15:00 | 1968.10 | 2025-03-28 10:15:00 | 2013.25 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2025-03-21 11:15:00 | 1969.80 | 2025-03-28 10:15:00 | 2013.25 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-03-21 12:00:00 | 1965.80 | 2025-03-28 10:15:00 | 2013.25 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-04-01 11:30:00 | 1967.80 | 2025-04-04 13:15:00 | 1997.55 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-04-01 12:30:00 | 1965.05 | 2025-04-04 13:15:00 | 1997.55 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-04-02 12:15:00 | 1971.35 | 2025-04-08 11:15:00 | 1997.35 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-04-02 13:30:00 | 1971.40 | 2025-04-08 11:15:00 | 1997.35 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-04-04 09:15:00 | 1969.40 | 2025-04-08 12:15:00 | 2014.35 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-04-04 10:15:00 | 1968.75 | 2025-04-08 12:15:00 | 2014.35 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1910.00 | 2025-04-08 12:15:00 | 2014.35 | STOP_HIT | 1.00 | -5.46% |
| SELL | retest2 | 2025-04-08 09:30:00 | 1968.50 | 2025-04-08 12:15:00 | 2014.35 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-04-09 14:15:00 | 2007.70 | 2025-04-15 10:15:00 | 2026.30 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-05-13 15:15:00 | 2070.10 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-16 10:45:00 | 2054.50 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-05-16 13:00:00 | 2053.80 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-19 11:15:00 | 2060.10 | 2025-05-22 09:15:00 | 2017.90 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-06 13:30:00 | 2067.30 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-10 14:15:00 | 2064.20 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-06-10 14:45:00 | 2071.00 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-11 11:00:00 | 2065.00 | 2025-06-13 09:15:00 | 2037.70 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-18 14:00:00 | 1994.30 | 2025-07-18 15:15:00 | 2009.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1987.80 | 2025-07-28 13:15:00 | 2006.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1994.50 | 2025-08-18 11:15:00 | 1894.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1994.50 | 2025-09-01 12:15:00 | 1795.05 | TARGET_HIT | 0.50 | 10.00% |
