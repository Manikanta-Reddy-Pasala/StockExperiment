# Narayana Hrudayalaya Ltd. (NH)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1820.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 22 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 21
- **Target hits / Stop hits / Partials:** 1 / 21 / 5
- **Avg / median % per leg:** -0.72% / -1.55%
- **Sum % (uncompounded):** -19.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.47% | -22.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 0 | 0.0% | 0 | 9 | 0 | -2.47% | -22.3% |
| SELL (all) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.15% | 2.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 18 | 6 | 33.3% | 1 | 12 | 5 | 0.15% | 2.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 6 | 22.2% | 1 | 21 | 5 | -0.72% | -19.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1817.50 | 1869.56 | 1869.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1803.60 | 1865.56 | 1867.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 13:15:00 | 1836.00 | 1835.15 | 1850.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-04 13:45:00 | 1833.10 | 1835.15 | 1850.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1806.70 | 1807.64 | 1830.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 1798.10 | 1807.60 | 1830.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:15:00 | 1800.00 | 1807.60 | 1830.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:00:00 | 1794.50 | 1807.46 | 1829.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1795.80 | 1807.34 | 1829.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1708.19 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1710.00 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1704.77 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1706.01 | 1786.36 | 1814.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1804.40 | 1774.90 | 1803.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 09:15:00 | 1804.40 | 1774.90 | 1803.39 | SL hit (close>ema200) qty=0.50 sl=1774.90 alert=retest2 |

### Cycle 2 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1953.50 | 1796.49 | 1796.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 1996.70 | 1801.44 | 1798.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1893.80 | 1893.99 | 1858.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 13:00:00 | 1893.80 | 1893.99 | 1858.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1860.90 | 1893.85 | 1860.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 1860.90 | 1893.85 | 1860.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1857.20 | 1893.48 | 1860.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 1858.70 | 1893.48 | 1860.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1858.40 | 1893.13 | 1860.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1886.30 | 1892.78 | 1860.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:45:00 | 1863.20 | 1889.74 | 1864.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 13:15:00 | 1838.60 | 1888.91 | 1864.19 | SL hit (close<static) qty=1.00 sl=1850.00 alert=retest2 |

### Cycle 3 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1723.50 | 1866.96 | 1867.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1713.60 | 1863.97 | 1865.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 1804.70 | 1803.23 | 1829.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 1804.70 | 1803.23 | 1829.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1836.90 | 1803.34 | 1828.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:45:00 | 1843.00 | 1803.34 | 1828.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1842.10 | 1803.73 | 1828.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 1842.20 | 1803.73 | 1828.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 1820.40 | 1812.58 | 1831.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 1803.40 | 1812.67 | 1831.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 1815.40 | 1812.70 | 1830.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 1833.00 | 1812.90 | 1830.99 | SL hit (close>static) qty=1.00 sl=1831.20 alert=retest2 |

### Cycle 4 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 1865.30 | 1767.05 | 1767.04 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-17 12:30:00 | 1798.10 | 2025-09-26 09:15:00 | 1708.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 13:15:00 | 1800.00 | 2025-09-26 09:15:00 | 1710.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 10:00:00 | 1794.50 | 2025-09-26 09:15:00 | 1704.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1795.80 | 2025-09-26 09:15:00 | 1706.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 12:30:00 | 1798.10 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.35% |
| SELL | retest2 | 2025-09-17 13:15:00 | 1800.00 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.24% |
| SELL | retest2 | 2025-09-18 10:00:00 | 1794.50 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.55% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1795.80 | 2025-10-06 09:15:00 | 1804.40 | STOP_HIT | 0.50 | -0.48% |
| SELL | retest2 | 2025-10-31 09:15:00 | 1767.80 | 2025-11-04 09:15:00 | 1835.70 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2025-11-03 13:15:00 | 1775.00 | 2025-11-04 09:15:00 | 1835.70 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-11-11 09:30:00 | 1762.90 | 2025-11-13 11:15:00 | 1794.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-11-13 12:15:00 | 1781.10 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -7.91% |
| SELL | retest2 | 2025-11-13 15:15:00 | 1773.00 | 2025-11-17 09:15:00 | 1921.90 | STOP_HIT | 1.00 | -8.40% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1886.30 | 2025-12-18 13:15:00 | 1838.60 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-12-18 11:45:00 | 1863.20 | 2025-12-18 13:15:00 | 1838.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-12-19 10:00:00 | 1873.80 | 2025-12-30 09:15:00 | 1851.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-12-26 15:15:00 | 1863.00 | 2025-12-30 10:15:00 | 1808.00 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2025-12-29 14:15:00 | 1866.30 | 2025-12-30 10:15:00 | 1808.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-12-30 14:30:00 | 1868.90 | 2025-12-30 15:15:00 | 1840.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-31 10:30:00 | 1873.60 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-01-01 09:30:00 | 1870.00 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2026-01-14 09:15:00 | 1933.00 | 2026-01-20 09:15:00 | 1831.20 | STOP_HIT | 1.00 | -5.27% |
| SELL | retest2 | 2026-02-16 09:15:00 | 1803.40 | 2026-02-16 10:15:00 | 1833.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-16 10:00:00 | 1815.40 | 2026-02-16 10:15:00 | 1833.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1814.20 | 2026-02-25 11:15:00 | 1862.20 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-09 09:15:00 | 1708.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1797.90 | 2026-03-16 12:15:00 | 1618.11 | TARGET_HIT | 0.50 | 10.00% |
