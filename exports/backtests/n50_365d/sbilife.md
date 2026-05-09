# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1871.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 21 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 11 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / Stop hits / Partials:** 0 / 11 / 1
- **Avg / median % per leg:** -0.81% / -1.48%
- **Sum % (uncompounded):** -9.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.83% | -14.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.83% | -14.6% |
| SELL (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.23% | 4.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.23% | 4.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.81% | -9.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1786.70 | 1809.18 | 1809.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1778.00 | 1808.87 | 1809.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1808.20 | 1805.73 | 1807.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1810.70 | 1805.78 | 1807.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1810.70 | 1805.78 | 1807.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1813.00 | 1805.86 | 1807.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1822.20 | 1805.86 | 1807.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1828.00 | 1806.08 | 1807.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 1828.50 | 1806.08 | 1807.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1808.10 | 1806.97 | 1808.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1802.20 | 1806.97 | 1808.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1821.20 | 1807.11 | 1808.07 | SL hit (close>static) qty=1.00 sl=1812.10 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1809.08 | 1809.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.18 | 1815.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.61 | 1916.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1963.70 | 1964.61 | 1916.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.62 | 2010.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 2030.00 | 2043.21 | 2009.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 2032.00 | 2042.90 | 2010.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:00:00 | 2029.80 | 2042.90 | 2010.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.05 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.05 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.05 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 2047.70 | 2030.11 | 2009.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 13:15:00 | 2010.70 | 2030.45 | 2009.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:00:00 | 2010.70 | 2030.45 | 2009.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 2000.50 | 2030.16 | 2009.73 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 2000.50 | 2030.16 | 2009.73 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-03 15:00:00 | 2000.50 | 2030.16 | 2009.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 2000.00 | 2029.86 | 2009.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:15:00 | 2011.10 | 2029.86 | 2009.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 2012.70 | 2029.64 | 2010.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 2009.40 | 2029.64 | 2010.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2014.30 | 2029.27 | 2010.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 2012.90 | 2029.27 | 2010.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2007.90 | 2029.06 | 2010.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 2007.90 | 2029.06 | 2010.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1985.70 | 2028.63 | 2010.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1985.70 | 2028.63 | 2010.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 1979.60 | 2028.14 | 2010.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:00:00 | 1979.60 | 2028.14 | 2010.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 2000.70 | 2026.31 | 2010.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 2000.70 | 2026.31 | 2010.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2005.10 | 2026.10 | 2010.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 12:45:00 | 2012.00 | 2025.76 | 2010.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:30:00 | 2010.00 | 2025.14 | 2011.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 2011.20 | 2024.88 | 2011.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 10:45:00 | 2011.80 | 2046.23 | 2028.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 2031.40 | 2044.85 | 2027.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:15:00 | 1970.60 | 2044.85 | 2027.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 1972.00 | 2044.12 | 2027.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1972.00 | 2044.12 | 2027.59 | SL hit (close<static) qty=1.00 sl=1999.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1972.00 | 2044.12 | 2027.59 | SL hit (close<static) qty=1.00 sl=1999.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1972.00 | 2044.12 | 2027.59 | SL hit (close<static) qty=1.00 sl=1999.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 09:15:00 | 1972.00 | 2044.12 | 2027.59 | SL hit (close<static) qty=1.00 sl=1999.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.00 | 2013.11 | 2013.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2012.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1914.90 | 1898.14 | 1942.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 1942.30 | 1899.91 | 1940.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1946.00 | 1900.37 | 1940.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 1946.00 | 1900.37 | 1940.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1942.90 | 1900.80 | 1940.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 1933.00 | 1901.10 | 1940.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1960.00 | 1903.55 | 1939.71 | SL hit (close>static) qty=1.00 sl=1950.60 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 1920.00 | 1920.14 | 1943.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:15:00 | 1824.00 | 1914.98 | 1939.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1874.70 | 1872.19 | 1908.01 | SL hit (close>ema200) qty=0.50 sl=1872.19 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-13 09:15:00 | 1802.20 | 2025-10-13 09:15:00 | 1821.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-27 09:45:00 | 2030.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 11:30:00 | 2032.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-27 12:00:00 | 2029.80 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2047.70 | 2026-02-03 14:15:00 | 2000.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-10 12:45:00 | 2012.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-12 10:30:00 | 2010.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-12 12:45:00 | 2011.20 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-02 10:45:00 | 2011.80 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-10 12:30:00 | 1933.00 | 2026-04-15 09:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-04-23 11:15:00 | 1824.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-05-07 09:15:00 | 1874.70 | STOP_HIT | 0.50 | 2.36% |
