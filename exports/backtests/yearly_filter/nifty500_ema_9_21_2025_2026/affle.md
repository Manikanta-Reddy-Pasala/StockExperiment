# Affle 3i Ltd. (AFFLE)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 1510.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 26 |
| ALERT3 | 130 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 71 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 85 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 50
- **Target hits / Stop hits / Partials:** 3 / 72 / 10
- **Avg / median % per leg:** 1.08% / -0.36%
- **Sum % (uncompounded):** 91.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 8 | 23.5% | 3 | 31 | 0 | 0.28% | 9.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.86% | -1.9% |
| BUY @ 3rd Alert (retest2) | 33 | 8 | 24.2% | 3 | 30 | 0 | 0.35% | 11.5% |
| SELL (all) | 51 | 27 | 52.9% | 0 | 41 | 10 | 1.61% | 82.2% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 2.28% | 6.8% |
| SELL @ 3rd Alert (retest2) | 48 | 24 | 50.0% | 0 | 38 | 10 | 1.57% | 75.3% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 1.24% | 5.0% |
| retest2 (combined) | 81 | 32 | 39.5% | 3 | 68 | 10 | 1.07% | 86.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1593.50 | 1541.20 | 1534.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 1610.60 | 1595.01 | 1579.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 09:15:00 | 1696.00 | 1700.61 | 1674.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:30:00 | 1688.30 | 1700.61 | 1674.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1699.90 | 1694.60 | 1683.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:45:00 | 1710.00 | 1695.71 | 1689.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 1721.10 | 1705.28 | 1699.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 1698.00 | 1716.92 | 1717.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 09:15:00 | 1698.00 | 1716.92 | 1717.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 12:15:00 | 1686.00 | 1703.47 | 1710.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 1704.90 | 1687.83 | 1696.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 12:45:00 | 1702.00 | 1687.83 | 1696.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1710.50 | 1692.36 | 1697.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 1709.00 | 1692.36 | 1697.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 15:15:00 | 1739.00 | 1708.56 | 1704.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1750.40 | 1725.09 | 1713.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 13:15:00 | 1725.00 | 1725.90 | 1716.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 14:00:00 | 1725.00 | 1725.90 | 1716.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1710.90 | 1727.00 | 1720.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1710.90 | 1727.00 | 1720.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1712.20 | 1724.04 | 1719.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:00:00 | 1712.20 | 1724.04 | 1719.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 1697.80 | 1716.48 | 1716.92 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 15:15:00 | 1760.00 | 1722.63 | 1719.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 1780.00 | 1758.97 | 1741.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1778.00 | 1787.86 | 1765.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:00:00 | 1778.00 | 1787.86 | 1765.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1793.40 | 1797.99 | 1790.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1793.30 | 1797.99 | 1790.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1787.50 | 1795.89 | 1790.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 1787.50 | 1795.89 | 1790.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1793.80 | 1795.47 | 1790.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1808.30 | 1795.47 | 1790.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 11:15:00 | 1799.50 | 1795.40 | 1791.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 1795.80 | 1793.96 | 1791.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 15:00:00 | 1800.00 | 1793.96 | 1791.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-10 09:15:00 | 1979.45 | 1850.02 | 1824.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 1885.90 | 1892.67 | 1893.50 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 14:15:00 | 1935.00 | 1900.72 | 1897.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-13 15:15:00 | 1946.00 | 1909.78 | 1901.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1905.40 | 1908.90 | 1901.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 1889.20 | 1908.90 | 1901.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1900.00 | 1907.12 | 1901.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 1900.00 | 1907.12 | 1901.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1902.10 | 1906.12 | 1901.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 1934.50 | 1910.88 | 1904.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 10:00:00 | 1910.20 | 1929.70 | 1924.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1901.10 | 1919.69 | 1920.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1901.10 | 1919.69 | 1920.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 13:15:00 | 1897.50 | 1912.98 | 1917.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1909.10 | 1905.64 | 1910.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:00:00 | 1909.10 | 1905.64 | 1910.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1901.10 | 1904.73 | 1909.63 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1927.50 | 1911.43 | 1909.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 10:15:00 | 1973.80 | 1945.93 | 1932.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1991.60 | 1996.39 | 1983.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:15:00 | 2011.60 | 1996.39 | 1983.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1987.90 | 1995.49 | 1985.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1987.90 | 1995.49 | 1985.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1987.90 | 1993.98 | 1985.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 1984.80 | 1993.98 | 1985.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1974.10 | 1990.00 | 1984.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 1974.10 | 1990.00 | 1984.53 | SL hit (close<ema400) qty=1.00 sl=1984.53 alert=retest1 |

### Cycle 10 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 1965.00 | 1979.59 | 1980.71 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 2008.00 | 1982.96 | 1981.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 2022.00 | 1990.77 | 1984.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 10:15:00 | 2024.70 | 2026.30 | 2010.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:00:00 | 2024.70 | 2026.30 | 2010.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2012.90 | 2022.14 | 2013.33 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 1992.50 | 2006.39 | 2008.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 15:15:00 | 1982.10 | 1996.67 | 2002.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1976.10 | 1961.13 | 1980.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1976.10 | 1961.13 | 1980.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1979.90 | 1964.88 | 1980.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 1985.00 | 1964.88 | 1980.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1972.40 | 1966.39 | 1979.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1994.70 | 1966.39 | 1979.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1983.50 | 1969.81 | 1979.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1997.00 | 1969.81 | 1979.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1987.80 | 1973.41 | 1980.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1987.80 | 1973.41 | 1980.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1979.40 | 1976.30 | 1980.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 13:30:00 | 1977.00 | 1977.80 | 1980.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:30:00 | 1974.50 | 1980.08 | 1981.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:30:00 | 1975.60 | 1980.65 | 1981.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1973.40 | 1970.85 | 1975.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1960.00 | 1968.68 | 1973.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 1970.00 | 1968.68 | 1973.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1990.00 | 1967.52 | 1971.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 15:00:00 | 1990.00 | 1967.52 | 1971.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1975.00 | 1969.02 | 1971.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1976.80 | 1969.02 | 1971.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1987.90 | 1972.79 | 1972.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 2000.00 | 1972.79 | 1972.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 10:15:00 | 1982.70 | 1974.77 | 1973.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1982.70 | 1974.77 | 1973.77 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 1968.10 | 1972.52 | 1972.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 15:15:00 | 1957.10 | 1968.95 | 1971.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1975.00 | 1970.16 | 1971.47 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 1974.80 | 1972.63 | 1972.45 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 13:15:00 | 1964.70 | 1971.04 | 1971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 1953.00 | 1965.22 | 1968.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 1917.00 | 1910.28 | 1924.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 1917.00 | 1910.28 | 1924.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1906.00 | 1907.95 | 1918.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 1899.70 | 1906.30 | 1916.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 1804.71 | 1828.20 | 1852.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 13:15:00 | 1820.00 | 1806.91 | 1829.40 | SL hit (close>ema200) qty=0.50 sl=1806.91 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 1894.00 | 1838.15 | 1838.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 13:15:00 | 1922.60 | 1868.82 | 1853.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 1958.10 | 1960.85 | 1925.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 15:15:00 | 1940.00 | 1957.90 | 1939.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 1940.00 | 1957.90 | 1939.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:30:00 | 1972.40 | 1960.93 | 1944.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 12:15:00 | 1970.70 | 1962.62 | 1946.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 10:15:00 | 1916.50 | 1944.13 | 1944.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 1916.50 | 1944.13 | 1944.39 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 1950.00 | 1939.38 | 1938.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 1960.00 | 1947.90 | 1943.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1946.20 | 1947.56 | 1943.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:00:00 | 1946.20 | 1947.56 | 1943.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1954.60 | 1948.97 | 1944.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:30:00 | 1956.10 | 1948.97 | 1944.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 1964.90 | 1952.15 | 1946.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1969.70 | 1948.50 | 1947.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 13:00:00 | 1974.60 | 1962.74 | 1955.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:15:00 | 1968.80 | 1965.74 | 1959.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:00:00 | 1970.80 | 1975.64 | 1969.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 1972.30 | 1974.97 | 1969.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:30:00 | 1972.30 | 1974.97 | 1969.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 1980.90 | 1976.16 | 1970.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 12:30:00 | 1987.50 | 1977.63 | 1971.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1985.00 | 1979.04 | 1972.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 1984.40 | 1979.25 | 1973.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1986.00 | 1978.40 | 1973.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1988.40 | 1980.40 | 1974.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 11:15:00 | 1970.00 | 1982.57 | 1980.61 | SL hit (close<static) qty=1.00 sl=1970.20 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1969.00 | 1977.83 | 1978.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1959.00 | 1972.16 | 1975.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1950.00 | 1947.32 | 1958.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:45:00 | 1945.00 | 1947.32 | 1958.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 1963.80 | 1951.57 | 1956.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 15:00:00 | 1963.80 | 1951.57 | 1956.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 15:15:00 | 1965.00 | 1954.25 | 1957.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1942.60 | 1954.25 | 1957.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 10:00:00 | 1954.90 | 1952.01 | 1954.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 15:15:00 | 1912.70 | 1905.44 | 1905.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 1912.70 | 1905.44 | 1905.03 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 1897.00 | 1903.80 | 1904.64 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1931.20 | 1909.28 | 1907.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 1935.30 | 1914.49 | 1909.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1912.20 | 1916.61 | 1911.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1912.20 | 1916.61 | 1911.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 1909.80 | 1915.25 | 1911.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:45:00 | 1909.00 | 1915.25 | 1911.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1902.70 | 1912.74 | 1910.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 1913.00 | 1912.74 | 1910.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1906.00 | 1911.39 | 1910.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1911.60 | 1911.39 | 1910.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 1912.70 | 1911.81 | 1910.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 11:15:00 | 1900.00 | 1909.82 | 1909.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1900.00 | 1909.82 | 1909.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1893.80 | 1906.61 | 1908.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1907.70 | 1905.66 | 1907.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1917.70 | 1905.66 | 1907.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1911.30 | 1906.79 | 1907.75 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 10:15:00 | 1925.00 | 1910.43 | 1909.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 1932.10 | 1916.33 | 1912.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 2023.70 | 2042.73 | 2010.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 2023.70 | 2042.73 | 2010.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2074.30 | 2083.38 | 2064.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 2074.30 | 2083.38 | 2064.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2072.10 | 2079.78 | 2066.25 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 11:15:00 | 2055.60 | 2062.71 | 2063.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 2048.40 | 2059.84 | 2061.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 2065.00 | 2060.88 | 2062.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:15:00 | 2064.40 | 2060.88 | 2062.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 2069.00 | 2062.50 | 2062.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 15:00:00 | 2069.00 | 2062.50 | 2062.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 2064.10 | 2062.82 | 2062.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 2135.00 | 2077.26 | 2069.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 2127.60 | 2128.88 | 2107.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 10:45:00 | 2136.30 | 2128.88 | 2107.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 2098.70 | 2121.97 | 2107.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 2098.70 | 2121.97 | 2107.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2105.10 | 2118.59 | 2107.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 2130.00 | 2113.36 | 2106.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 12:15:00 | 2114.20 | 2131.35 | 2132.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 12:15:00 | 2114.20 | 2131.35 | 2132.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 13:15:00 | 2101.20 | 2125.32 | 2129.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 2068.20 | 2064.07 | 2086.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:15:00 | 2023.70 | 2053.66 | 2070.48 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:00:00 | 2020.40 | 2047.01 | 2065.92 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 10:45:00 | 2019.20 | 2041.61 | 2061.75 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 1969.90 | 1956.84 | 1973.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 1969.90 | 1956.84 | 1973.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 1975.00 | 1960.47 | 1973.38 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 1975.00 | 1960.47 | 1973.38 | SL hit (close>ema400) qty=1.00 sl=1973.38 alert=retest1 |

### Cycle 29 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1942.00 | 1929.90 | 1929.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 1947.80 | 1933.64 | 1931.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1958.90 | 1961.72 | 1952.08 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 10:15:00 | 1934.10 | 1951.91 | 1952.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1916.30 | 1944.79 | 1948.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1941.20 | 1933.69 | 1940.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1941.20 | 1933.69 | 1940.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1931.80 | 1933.31 | 1939.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:45:00 | 1929.00 | 1931.53 | 1938.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1964.70 | 1930.50 | 1934.61 | SL hit (close>static) qty=1.00 sl=1948.70 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 1948.30 | 1937.50 | 1937.29 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1927.60 | 1938.00 | 1938.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1914.20 | 1931.16 | 1934.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 09:15:00 | 1890.00 | 1889.04 | 1901.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:45:00 | 1889.10 | 1889.04 | 1901.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1900.00 | 1892.63 | 1901.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:15:00 | 1907.00 | 1892.63 | 1901.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1906.80 | 1895.46 | 1901.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1908.00 | 1895.46 | 1901.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1903.00 | 1896.97 | 1901.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 1908.40 | 1896.97 | 1901.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1896.20 | 1896.82 | 1901.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:45:00 | 1933.00 | 1896.82 | 1901.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1886.30 | 1894.42 | 1899.31 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1919.20 | 1900.23 | 1897.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 1925.30 | 1905.24 | 1900.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 14:15:00 | 1920.80 | 1925.23 | 1914.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 1920.80 | 1925.23 | 1914.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1918.50 | 1923.88 | 1915.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1893.40 | 1923.88 | 1915.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1890.00 | 1917.11 | 1912.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:45:00 | 1888.70 | 1917.11 | 1912.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1894.00 | 1912.49 | 1911.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 11:15:00 | 1897.50 | 1912.49 | 1911.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1895.70 | 1909.13 | 1909.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1895.70 | 1909.13 | 1909.70 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 1932.00 | 1912.09 | 1910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 1939.60 | 1917.59 | 1912.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1928.00 | 1931.42 | 1922.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 1928.00 | 1931.42 | 1922.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1925.00 | 1932.37 | 1926.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1918.80 | 1932.37 | 1926.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1898.00 | 1925.49 | 1924.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:45:00 | 1897.00 | 1925.49 | 1924.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1911.00 | 1922.60 | 1923.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 09:15:00 | 1826.90 | 1892.20 | 1907.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 1747.20 | 1738.43 | 1763.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 1747.20 | 1738.43 | 1763.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1741.80 | 1739.51 | 1759.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:30:00 | 1728.30 | 1736.63 | 1748.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 12:15:00 | 1764.00 | 1744.88 | 1749.52 | SL hit (close>static) qty=1.00 sl=1760.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1777.00 | 1754.76 | 1753.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 15:15:00 | 1779.00 | 1759.61 | 1755.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1755.10 | 1758.71 | 1755.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1755.10 | 1758.71 | 1755.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1749.00 | 1756.77 | 1755.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1749.00 | 1756.77 | 1755.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1746.00 | 1754.61 | 1754.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 1746.40 | 1754.61 | 1754.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 1744.90 | 1752.92 | 1753.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 1737.50 | 1749.84 | 1752.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 13:15:00 | 1729.40 | 1728.17 | 1734.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:30:00 | 1729.30 | 1728.17 | 1734.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1687.30 | 1720.25 | 1729.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1676.60 | 1700.45 | 1712.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1676.30 | 1688.23 | 1699.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 13:15:00 | 1717.60 | 1706.11 | 1705.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1717.60 | 1706.11 | 1705.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1720.20 | 1710.99 | 1707.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 1708.60 | 1711.32 | 1708.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 1708.60 | 1711.32 | 1708.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 1717.10 | 1712.47 | 1709.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:45:00 | 1707.20 | 1712.47 | 1709.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 1707.70 | 1713.32 | 1711.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:45:00 | 1701.40 | 1713.32 | 1711.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 1693.50 | 1709.35 | 1709.58 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 1718.90 | 1710.13 | 1709.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 15:15:00 | 1720.00 | 1712.11 | 1710.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1701.80 | 1710.05 | 1709.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1701.80 | 1710.05 | 1709.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 1700.00 | 1708.04 | 1708.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 14:15:00 | 1683.50 | 1697.92 | 1703.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 1686.80 | 1678.94 | 1686.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 1686.80 | 1678.94 | 1686.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 1678.90 | 1678.93 | 1686.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 1676.80 | 1678.78 | 1685.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 1674.80 | 1677.48 | 1683.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 11:15:00 | 1672.10 | 1678.75 | 1682.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1701.60 | 1676.83 | 1677.90 | SL hit (close>static) qty=1.00 sl=1688.70 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 12:15:00 | 1709.50 | 1683.36 | 1680.77 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 1656.80 | 1679.34 | 1680.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1653.20 | 1674.11 | 1678.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1645.10 | 1639.10 | 1648.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 15:00:00 | 1645.10 | 1639.10 | 1648.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1643.00 | 1639.88 | 1647.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 1647.30 | 1639.88 | 1647.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1634.10 | 1638.72 | 1646.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 10:15:00 | 1631.00 | 1638.72 | 1646.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 11:15:00 | 1628.00 | 1637.74 | 1645.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:00:00 | 1623.40 | 1630.36 | 1637.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1643.40 | 1626.67 | 1625.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1643.40 | 1626.67 | 1625.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1655.20 | 1634.51 | 1629.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1634.60 | 1638.46 | 1634.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 15:00:00 | 1634.60 | 1638.46 | 1634.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1626.60 | 1636.09 | 1633.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:15:00 | 1625.00 | 1636.09 | 1633.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 1631.90 | 1635.25 | 1633.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1622.30 | 1635.25 | 1633.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1690.80 | 1704.65 | 1695.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:00:00 | 1690.80 | 1704.65 | 1695.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1682.90 | 1700.30 | 1693.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 1686.00 | 1700.30 | 1693.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 1660.00 | 1687.52 | 1689.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1639.70 | 1673.86 | 1682.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1670.10 | 1667.74 | 1675.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 1670.10 | 1667.74 | 1675.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1677.00 | 1670.63 | 1675.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 1682.30 | 1670.63 | 1675.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 1674.10 | 1671.32 | 1675.38 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1721.40 | 1684.69 | 1680.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 1745.00 | 1696.75 | 1686.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 11:15:00 | 1698.90 | 1700.54 | 1691.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 11:45:00 | 1702.20 | 1700.54 | 1691.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1769.20 | 1771.48 | 1753.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:00:00 | 1783.30 | 1771.55 | 1759.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:00:00 | 1776.70 | 1765.20 | 1762.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 1776.70 | 1775.27 | 1769.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 1777.40 | 1775.64 | 1770.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1791.10 | 1782.50 | 1775.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 1781.90 | 1782.50 | 1775.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1779.70 | 1786.61 | 1781.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:45:00 | 1776.90 | 1786.61 | 1781.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1786.50 | 1786.59 | 1781.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1780.00 | 1786.59 | 1781.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1788.30 | 1786.93 | 1782.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:45:00 | 1785.50 | 1786.93 | 1782.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1783.10 | 1786.17 | 1782.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1783.10 | 1786.17 | 1782.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1779.00 | 1784.73 | 1782.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:15:00 | 1762.10 | 1784.73 | 1782.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1771.30 | 1782.05 | 1781.14 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-02 10:15:00 | 1768.70 | 1779.38 | 1780.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 1768.70 | 1779.38 | 1780.01 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 1786.40 | 1781.38 | 1780.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 1803.70 | 1785.53 | 1782.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 13:15:00 | 1813.50 | 1814.03 | 1804.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 1813.50 | 1814.03 | 1804.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1807.20 | 1813.02 | 1806.58 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1800.50 | 1804.07 | 1804.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 1791.00 | 1801.46 | 1803.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1747.00 | 1738.63 | 1757.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1733.60 | 1738.23 | 1753.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 1735.20 | 1736.64 | 1751.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 1734.50 | 1734.11 | 1747.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 1729.00 | 1731.45 | 1738.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1726.90 | 1730.15 | 1736.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 1720.00 | 1727.66 | 1734.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 1708.10 | 1723.20 | 1729.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1646.92 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1648.44 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1647.77 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 13:15:00 | 1642.55 | 1668.98 | 1690.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1634.00 | 1651.71 | 1676.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 1622.69 | 1651.71 | 1676.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-21 15:15:00 | 1623.00 | 1622.96 | 1648.45 | SL hit (close>ema200) qty=0.50 sl=1622.96 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1567.00 | 1555.28 | 1554.41 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 14:15:00 | 1539.40 | 1552.11 | 1553.05 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 1560.20 | 1554.83 | 1554.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 1639.00 | 1571.66 | 1561.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 1585.10 | 1591.63 | 1577.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 1572.40 | 1591.63 | 1577.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 1590.30 | 1591.37 | 1578.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 1616.20 | 1587.43 | 1580.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 1646.10 | 1650.06 | 1650.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 15:15:00 | 1646.10 | 1650.06 | 1650.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1621.40 | 1644.33 | 1647.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 1589.50 | 1581.34 | 1595.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 1589.50 | 1581.34 | 1595.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 1586.30 | 1583.03 | 1592.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 1585.10 | 1583.03 | 1592.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1588.40 | 1584.11 | 1591.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 1588.40 | 1584.11 | 1591.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1566.70 | 1580.62 | 1589.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 1564.50 | 1577.54 | 1587.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 1565.20 | 1577.54 | 1587.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 1561.90 | 1574.03 | 1584.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1486.27 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1486.94 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 1483.81 | 1509.34 | 1529.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-23 09:15:00 | 1509.10 | 1495.34 | 1509.99 | SL hit (close>ema200) qty=0.50 sl=1495.34 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 14:15:00 | 1403.60 | 1375.73 | 1371.98 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 13:15:00 | 1353.10 | 1369.19 | 1370.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 15:15:00 | 1347.00 | 1357.67 | 1362.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 11:15:00 | 1367.90 | 1354.79 | 1359.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 11:30:00 | 1372.50 | 1354.79 | 1359.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 1386.90 | 1361.22 | 1362.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 1386.90 | 1361.22 | 1362.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 13:15:00 | 1386.00 | 1366.17 | 1364.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 14:15:00 | 1392.10 | 1371.36 | 1366.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 1376.90 | 1377.57 | 1371.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:45:00 | 1377.20 | 1377.57 | 1371.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 1396.70 | 1381.40 | 1374.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 1400.00 | 1386.09 | 1377.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 1400.00 | 1397.72 | 1394.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 14:00:00 | 1401.00 | 1398.38 | 1394.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 1371.80 | 1393.06 | 1392.56 | SL hit (close<static) qty=1.00 sl=1372.70 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 1380.00 | 1390.45 | 1391.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1335.90 | 1379.54 | 1386.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1301.40 | 1295.73 | 1312.21 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 1345.70 | 1324.30 | 1321.58 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1300.70 | 1320.52 | 1320.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1300.20 | 1309.70 | 1314.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 1311.50 | 1306.07 | 1311.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 1311.50 | 1306.07 | 1311.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1310.10 | 1306.88 | 1311.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1312.90 | 1306.88 | 1311.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 1293.90 | 1304.28 | 1309.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 14:15:00 | 1286.80 | 1302.03 | 1308.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 1307.80 | 1290.20 | 1289.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1307.80 | 1290.20 | 1289.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1375.20 | 1309.25 | 1298.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 11:15:00 | 1432.30 | 1432.67 | 1400.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:00:00 | 1432.30 | 1432.67 | 1400.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1435.90 | 1447.36 | 1431.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 1435.90 | 1447.36 | 1431.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 1438.00 | 1445.49 | 1431.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:15:00 | 1417.00 | 1445.49 | 1431.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1411.80 | 1438.75 | 1429.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 1411.80 | 1438.75 | 1429.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1414.40 | 1433.88 | 1428.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 1418.60 | 1433.88 | 1428.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 1419.00 | 1425.87 | 1425.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 1419.00 | 1425.87 | 1425.96 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1430.00 | 1426.52 | 1426.18 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-06 13:15:00 | 1418.50 | 1424.98 | 1425.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 09:15:00 | 1412.80 | 1420.70 | 1423.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 1421.00 | 1420.47 | 1422.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 11:45:00 | 1422.10 | 1420.47 | 1422.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 1412.10 | 1418.80 | 1421.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 14:15:00 | 1408.10 | 1417.46 | 1420.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 14:45:00 | 1404.40 | 1416.75 | 1420.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1406.90 | 1416.75 | 1420.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1433.60 | 1418.54 | 1420.35 | SL hit (close>static) qty=1.00 sl=1422.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 13:15:00 | 1415.70 | 1407.11 | 1406.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1422.20 | 1413.39 | 1409.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1418.20 | 1419.84 | 1416.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 11:45:00 | 1420.10 | 1419.84 | 1416.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1434.40 | 1422.75 | 1417.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 13:30:00 | 1452.30 | 1428.50 | 1420.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 11:15:00 | 1445.00 | 1453.04 | 1453.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 11:15:00 | 1445.00 | 1453.04 | 1453.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 1440.70 | 1446.71 | 1450.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1439.90 | 1417.00 | 1425.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1439.90 | 1417.00 | 1425.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1440.20 | 1421.64 | 1426.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 1440.00 | 1421.64 | 1426.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1456.00 | 1431.91 | 1430.75 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 09:15:00 | 1431.50 | 1437.39 | 1437.72 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 1442.20 | 1438.35 | 1438.12 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 1435.00 | 1437.98 | 1438.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 14:15:00 | 1430.60 | 1436.51 | 1437.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 1428.00 | 1424.54 | 1428.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 1426.00 | 1424.54 | 1428.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1424.90 | 1424.61 | 1428.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:30:00 | 1418.00 | 1420.55 | 1426.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 1417.30 | 1414.99 | 1421.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:30:00 | 1415.00 | 1417.23 | 1420.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:15:00 | 1417.70 | 1416.87 | 1418.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1423.00 | 1418.09 | 1418.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:45:00 | 1424.90 | 1418.09 | 1418.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1419.50 | 1418.38 | 1418.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-06 13:15:00 | 1422.70 | 1419.24 | 1419.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 13:15:00 | 1422.70 | 1419.24 | 1419.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1443.80 | 1424.15 | 1421.39 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 09:45:00 | 1710.00 | 2025-05-27 09:15:00 | 1698.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-23 09:30:00 | 1721.10 | 2025-05-27 09:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-06-06 09:15:00 | 1808.30 | 2025-06-10 09:15:00 | 1979.45 | TARGET_HIT | 1.00 | 9.46% |
| BUY | retest2 | 2025-06-06 11:15:00 | 1799.50 | 2025-06-10 09:15:00 | 1975.38 | TARGET_HIT | 1.00 | 9.77% |
| BUY | retest2 | 2025-06-06 14:30:00 | 1795.80 | 2025-06-10 09:15:00 | 1980.00 | TARGET_HIT | 1.00 | 10.26% |
| BUY | retest2 | 2025-06-06 15:00:00 | 1800.00 | 2025-06-13 12:15:00 | 1885.90 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-06-13 11:00:00 | 1892.00 | 2025-06-13 12:15:00 | 1885.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-06-16 14:00:00 | 1934.50 | 2025-06-18 11:15:00 | 1901.10 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-06-18 10:00:00 | 1910.20 | 2025-06-18 11:15:00 | 1901.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-07-01 09:15:00 | 2011.60 | 2025-07-01 12:15:00 | 1974.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-07-09 13:30:00 | 1977.00 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-07-09 14:30:00 | 1974.50 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-07-10 10:30:00 | 1975.60 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1973.40 | 2025-07-14 10:15:00 | 1982.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-07-22 11:00:00 | 1899.70 | 2025-07-25 14:15:00 | 1804.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 1899.70 | 2025-07-28 13:15:00 | 1820.00 | STOP_HIT | 0.50 | 4.20% |
| BUY | retest2 | 2025-08-01 10:30:00 | 1972.40 | 2025-08-04 10:15:00 | 1916.50 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-08-01 12:15:00 | 1970.70 | 2025-08-04 10:15:00 | 1916.50 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2025-08-08 09:15:00 | 1969.70 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-08-08 13:00:00 | 1974.60 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-08-11 10:15:00 | 1968.80 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-08-12 10:00:00 | 1970.80 | 2025-08-14 11:15:00 | 1970.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-08-12 12:30:00 | 1987.50 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1985.00 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-08-12 14:30:00 | 1984.40 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-08-13 09:15:00 | 1986.00 | 2025-08-14 13:15:00 | 1969.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1942.60 | 2025-09-01 15:15:00 | 1912.70 | STOP_HIT | 1.00 | 1.54% |
| SELL | retest2 | 2025-08-21 10:00:00 | 1954.90 | 2025-09-01 15:15:00 | 1912.70 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2025-09-04 09:15:00 | 1911.60 | 2025-09-04 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-09-04 09:45:00 | 1912.70 | 2025-09-04 11:15:00 | 1900.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-09-19 09:15:00 | 2130.00 | 2025-09-23 12:15:00 | 2114.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest1 | 2025-09-26 09:15:00 | 2023.70 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.41% |
| SELL | retest1 | 2025-09-26 10:00:00 | 2020.40 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.25% |
| SELL | retest1 | 2025-09-26 10:45:00 | 2019.20 | 2025-10-01 12:15:00 | 1975.00 | STOP_HIT | 1.00 | 2.19% |
| SELL | retest2 | 2025-10-03 09:15:00 | 1951.00 | 2025-10-09 09:15:00 | 1942.00 | STOP_HIT | 1.00 | 0.46% |
| SELL | retest2 | 2025-10-06 13:15:00 | 1957.50 | 2025-10-09 09:15:00 | 1942.00 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2025-10-15 12:45:00 | 1929.00 | 2025-10-16 09:15:00 | 1964.70 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-29 11:15:00 | 1897.50 | 2025-10-29 11:15:00 | 1895.70 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-12 09:30:00 | 1728.30 | 2025-11-12 12:15:00 | 1764.00 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-11-19 09:30:00 | 1676.60 | 2025-11-20 13:15:00 | 1717.60 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-11-20 09:30:00 | 1676.30 | 2025-11-20 13:15:00 | 1717.60 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-11-27 13:15:00 | 1676.80 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-27 14:30:00 | 1674.80 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-28 11:15:00 | 1672.10 | 2025-12-01 11:15:00 | 1701.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-12-05 10:15:00 | 1631.00 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-05 11:15:00 | 1628.00 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-08 10:00:00 | 1623.40 | 2025-12-09 14:15:00 | 1643.40 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-26 15:00:00 | 1783.30 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-30 10:00:00 | 1776.70 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-30 14:30:00 | 1776.70 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-31 09:45:00 | 1777.40 | 2026-01-02 10:15:00 | 1768.70 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1733.60 | 2026-01-20 13:15:00 | 1646.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1735.20 | 2026-01-20 13:15:00 | 1648.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 1734.50 | 2026-01-20 13:15:00 | 1647.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1729.00 | 2026-01-20 13:15:00 | 1642.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1720.00 | 2026-01-21 09:15:00 | 1634.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1708.10 | 2026-01-21 09:15:00 | 1622.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1733.60 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.38% |
| SELL | retest2 | 2026-01-13 12:30:00 | 1735.20 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.47% |
| SELL | retest2 | 2026-01-13 15:00:00 | 1734.50 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.43% |
| SELL | retest2 | 2026-01-14 15:15:00 | 1729.00 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 6.13% |
| SELL | retest2 | 2026-01-16 11:45:00 | 1720.00 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2026-01-19 09:15:00 | 1708.10 | 2026-01-21 15:15:00 | 1623.00 | STOP_HIT | 0.50 | 4.98% |
| BUY | retest2 | 2026-02-03 09:15:00 | 1616.20 | 2026-02-11 15:15:00 | 1646.10 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2026-02-17 13:30:00 | 1564.50 | 2026-02-20 09:15:00 | 1486.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1565.20 | 2026-02-20 09:15:00 | 1486.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1561.90 | 2026-02-20 09:15:00 | 1483.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:30:00 | 1564.50 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2026-02-17 14:00:00 | 1565.20 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-02-17 14:30:00 | 1561.90 | 2026-02-23 09:15:00 | 1509.10 | STOP_HIT | 0.50 | 3.38% |
| BUY | retest2 | 2026-03-10 15:00:00 | 1400.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-12 13:00:00 | 1400.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-03-12 14:00:00 | 1401.00 | 2026-03-12 14:15:00 | 1371.80 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-03-20 14:15:00 | 1286.80 | 2026-03-24 14:15:00 | 1307.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1418.60 | 2026-04-02 15:15:00 | 1419.00 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2026-04-07 14:15:00 | 1408.10 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-04-07 14:45:00 | 1404.40 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2026-04-07 15:15:00 | 1406.90 | 2026-04-08 09:15:00 | 1433.60 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1408.10 | 2026-04-13 13:15:00 | 1415.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1379.40 | 2026-04-13 13:15:00 | 1415.70 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2026-04-16 13:30:00 | 1452.30 | 2026-04-22 11:15:00 | 1445.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-05-04 10:30:00 | 1418.00 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-05-04 14:30:00 | 1417.30 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-05-05 10:30:00 | 1415.00 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-05-06 11:15:00 | 1417.70 | 2026-05-06 13:15:00 | 1422.70 | STOP_HIT | 1.00 | -0.35% |
