# BEML Ltd. (BEML)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1952.00
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
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 7
- **Target hits / Stop hits / Partials:** 2 / 11 / 4
- **Avg / median % per leg:** 1.79% / 1.51%
- **Sum % (uncompounded):** 30.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 16 | 9 | 56.2% | 1 | 11 | 4 | 1.28% | 20.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 9 | 56.2% | 1 | 11 | 4 | 1.28% | 20.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 17 | 10 | 58.8% | 2 | 11 | 4 | 1.79% | 30.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1732.50 | 1547.48 | 1547.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 09:15:00 | 1790.00 | 1566.41 | 1556.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 2206.85 | 2211.87 | 2077.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 14:00:00 | 2206.85 | 2211.87 | 2077.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 2093.15 | 2198.96 | 2090.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:00:00 | 2093.15 | 2198.96 | 2090.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 2089.55 | 2197.87 | 2090.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 11:30:00 | 2090.15 | 2197.87 | 2090.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 2096.50 | 2196.86 | 2090.33 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 1922.15 | 2040.72 | 2040.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 1913.30 | 2037.04 | 2038.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 2050.90 | 2023.70 | 2031.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 2050.90 | 2023.70 | 2031.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 2030.50 | 2023.76 | 2031.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 2023.40 | 2025.63 | 2032.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 10:00:00 | 2016.05 | 2025.53 | 2032.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 2062.95 | 2024.09 | 2031.28 | SL hit (close>static) qty=1.00 sl=2054.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 2062.95 | 2024.09 | 2031.28 | SL hit (close>static) qty=1.00 sl=2054.45 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 2024.00 | 2027.05 | 2032.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 2024.50 | 2027.56 | 2032.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 2033.50 | 2027.62 | 2032.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 2062.85 | 2027.62 | 2032.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2042.15 | 2027.76 | 2032.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 2030.60 | 2027.97 | 2032.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2086.70 | 2028.42 | 2032.53 | SL hit (close>static) qty=1.00 sl=2054.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2086.70 | 2028.42 | 2032.53 | SL hit (close>static) qty=1.00 sl=2054.45 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2086.70 | 2028.42 | 2032.53 | SL hit (close>static) qty=1.00 sl=2073.45 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 15:15:00 | 2187.50 | 2037.21 | 2036.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 2245.00 | 2084.24 | 2062.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 2096.35 | 2106.21 | 2077.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 2096.35 | 2106.21 | 2077.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 2081.20 | 2106.08 | 2078.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 11:00:00 | 2081.20 | 2106.08 | 2078.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 2051.45 | 2105.53 | 2077.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 2051.45 | 2105.53 | 2077.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 2049.60 | 2104.98 | 2077.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:45:00 | 2043.95 | 2104.98 | 2077.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 2061.60 | 2101.39 | 2076.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-30 14:30:00 | 2068.50 | 2100.75 | 2076.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-24 11:15:00 | 2275.35 | 2158.81 | 2121.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 2031.10 | 2110.15 | 2110.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 2011.00 | 2109.17 | 2109.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 09:15:00 | 1857.30 | 1809.05 | 1905.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-23 09:45:00 | 1868.70 | 1809.05 | 1905.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1894.50 | 1815.18 | 1902.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 1853.40 | 1823.60 | 1901.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:30:00 | 1867.30 | 1824.92 | 1898.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 14:45:00 | 1859.90 | 1830.78 | 1895.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 10:15:00 | 1860.70 | 1835.65 | 1894.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1760.73 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1773.93 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1766.90 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 1767.66 | 1835.25 | 1887.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1827.23 | 1878.92 | SL hit (close>ema200) qty=0.50 sl=1827.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1827.23 | 1878.92 | SL hit (close>ema200) qty=0.50 sl=1827.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1827.23 | 1878.92 | SL hit (close>ema200) qty=0.50 sl=1827.23 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 11:15:00 | 1831.80 | 1827.23 | 1878.92 | SL hit (close>ema200) qty=0.50 sl=1827.23 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1820.00 | 1779.51 | 1836.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1820.00 | 1779.51 | 1836.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1815.00 | 1780.57 | 1834.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:00:00 | 1788.30 | 1780.65 | 1834.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-01 12:15:00 | 1609.47 | 1779.68 | 1833.71 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 1785.50 | 1587.13 | 1637.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 11:00:00 | 1789.50 | 1649.56 | 1662.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1809.50 | 1675.32 | 1675.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1809.50 | 1675.32 | 1675.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 1809.50 | 1675.32 | 1675.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 1842.50 | 1676.98 | 1675.91 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-04 09:15:00 | 2023.40 | 2025-09-05 10:15:00 | 2062.95 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-04 10:00:00 | 2016.05 | 2025-09-05 10:15:00 | 2062.95 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-09-09 14:30:00 | 2024.00 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest2 | 2025-09-10 14:45:00 | 2024.50 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-09-11 11:30:00 | 2030.60 | 2025-09-12 09:15:00 | 2086.70 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-09-30 14:30:00 | 2068.50 | 2025-10-24 11:15:00 | 2275.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-29 15:00:00 | 1853.40 | 2026-01-12 09:15:00 | 1760.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1867.30 | 2026-01-12 09:15:00 | 1773.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-02 14:45:00 | 1859.90 | 2026-01-12 09:15:00 | 1766.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:15:00 | 1860.70 | 2026-01-12 09:15:00 | 1767.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 15:00:00 | 1853.40 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1867.30 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.90% |
| SELL | retest2 | 2026-01-02 14:45:00 | 1859.90 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.51% |
| SELL | retest2 | 2026-01-06 10:15:00 | 1860.70 | 2026-01-14 11:15:00 | 1831.80 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2026-02-01 11:00:00 | 1788.30 | 2026-02-01 12:15:00 | 1609.47 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-17 09:45:00 | 1785.50 | 2026-04-28 13:15:00 | 1809.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2026-04-24 11:00:00 | 1789.50 | 2026-04-28 13:15:00 | 1809.50 | STOP_HIT | 1.00 | -1.12% |
