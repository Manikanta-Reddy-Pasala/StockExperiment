# SBILIFE (SBILIFE)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
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
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 14 |
| PENDING_CANCEL | 6 |
| ENTRY1 | 3 |
| ENTRY2 | 5 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 12 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / Stop hits / Partials:** 0 / 8 / 4
- **Avg / median % per leg:** 2.11% / 3.57%
- **Sum % (uncompounded):** 25.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 6 | 60.0% | 0 | 7 | 3 | 1.85% | 18.5% |
| BUY @ 2nd Alert (retest1) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.27% | 25.6% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.78% | -7.1% |
| SELL (all) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.42% | 6.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 0 | 1 | 1 | 3.42% | 6.8% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 3 | 3 | 4.27% | 25.6% |
| retest2 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | -0.05% | -0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 1773.50 | 1818.09 | 1818.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 10:15:00 | 1771.70 | 1816.25 | 1817.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.20 | 1805.80 | 1811.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 1808.20 | 1805.80 | 1811.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1808.20 | 1805.80 | 1811.38 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 15:15:00 | 1839.00 | 1815.38 | 1815.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1852.70 | 1816.22 | 1815.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.62 | 1917.27 | EMA200 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2025-12-01 15:15:00 | 1974.00 | 1964.58 | 1919.11 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-12-02 09:15:00 | 1960.80 | 1964.55 | 1919.32 | ENTRY1 sustain failed after 1080m |
| Cross detected — sustain check pending | 2025-12-02 13:15:00 | 1975.40 | 1964.64 | 1920.26 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:15:00 | 1980.30 | 1964.79 | 1920.56 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 11:15:00 | 1976.80 | 1965.31 | 1921.70 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 12:15:00 | 1974.50 | 1965.40 | 1921.96 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-03 15:15:00 | 1974.00 | 1965.58 | 1922.69 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1976.60 | 1965.68 | 1922.96 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 12:15:00 | 2073.22 | 2008.73 | 1973.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 12:15:00 | 2075.43 | 2008.73 | 1973.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 2079.32 | 2011.25 | 1975.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 2047.10 | 2047.40 | 2008.12 | SL hit (close<ema200) qty=0.50 sl=2047.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 2047.10 | 2047.40 | 2008.12 | SL hit (close<ema200) qty=0.50 sl=2047.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 2047.10 | 2047.40 | 2008.12 | SL hit (close<ema200) qty=0.50 sl=2047.40 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.62 | 2010.38 | EMA400 retest candle locked (from upside) |
| Cross detected — sustain check pending | 2026-01-27 09:15:00 | 2030.30 | 2043.21 | 2010.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-27 10:15:00 | 2025.50 | 2043.03 | 2010.31 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-27 11:15:00 | 2029.80 | 2042.90 | 2010.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:15:00 | 2034.00 | 2042.81 | 2010.53 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.35 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-03 09:15:00 | 2051.70 | 2030.33 | 2009.56 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2049.70 | 2030.52 | 2009.76 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 2000.50 | 2030.16 | 2009.99 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 12:15:00 | 2040.80 | 2029.78 | 2010.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 2042.50 | 2029.90 | 2010.45 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 2007.90 | 2029.06 | 2010.97 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-09 10:15:00 | 2033.40 | 2026.68 | 2010.46 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-09 11:15:00 | 2025.40 | 2026.66 | 2010.53 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2031.30 | 2024.80 | 2011.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-13 10:15:00 | 2020.70 | 2024.76 | 2011.55 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-13 13:15:00 | 2031.90 | 2024.73 | 2011.73 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 14:15:00 | 2032.60 | 2024.81 | 2011.84 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2010.90 | 2046.48 | 2028.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-02 11:15:00 | 2005.00 | 2045.82 | 2028.14 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |

### Cycle 3 — SELL (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 14:15:00 | 1964.00 | 2013.61 | 2013.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2013.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.63 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1909.70 | 1901.92 | 1940.26 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-13 10:15:00 | 1918.60 | 1902.08 | 1940.15 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-13 15:15:00 | 1914.10 | 1902.98 | 1939.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-15 09:15:00 | 1960.00 | 1903.55 | 1939.77 | ENTRY2 sustain failed after 2520m |
| Cross detected — sustain check pending | 2026-04-21 09:15:00 | 1900.00 | 1919.94 | 1943.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 1910.00 | 1919.84 | 1943.64 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 11:15:00 | 1814.50 | 1914.98 | 1939.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 1874.70 | 1872.19 | 1908.05 | SL hit (close>ema200) qty=0.50 sl=1872.19 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-02 12:15:00 | 2073.22 | PARTIAL | 0.50 | 4.69% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.50 | 2026-01-02 12:15:00 | 2075.43 | PARTIAL | 0.50 | 5.11% |
| BUY | retest1 | 2025-12-04 09:15:00 | 1976.60 | 2026-01-05 09:15:00 | 2079.32 | PARTIAL | 0.50 | 5.20% |
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-20 14:15:00 | 2047.10 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.50 | 2026-01-20 14:15:00 | 2047.10 | STOP_HIT | 0.50 | 3.68% |
| BUY | retest1 | 2025-12-04 09:15:00 | 1976.60 | 2026-01-20 14:15:00 | 2047.10 | STOP_HIT | 0.50 | 3.57% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2034.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2049.70 | 2026-02-03 14:15:00 | 2000.50 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2042.50 | 2026-02-06 09:15:00 | 2007.90 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2032.60 | 2026-03-02 11:15:00 | 2005.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1910.00 | 2026-04-23 11:15:00 | 1814.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 10:15:00 | 1910.00 | 2026-05-07 09:15:00 | 1874.70 | STOP_HIT | 0.50 | 1.85% |
