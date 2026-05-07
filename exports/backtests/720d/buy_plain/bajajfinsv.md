# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1829.00
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 3 |
| ALERT3 | 7 |
| PENDING | 24 |
| PENDING_CANCEL | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 17 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 17
- **Target hits / Stop hits / Partials:** 0 / 21 / 2
- **Avg / median % per leg:** 0.39% / -1.42%
- **Sum % (uncompounded):** 8.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 6 | 26.1% | 0 | 21 | 2 | 0.39% | 8.9% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.31% | -17.2% |
| BUY @ 3rd Alert (retest2) | 19 | 6 | 31.6% | 0 | 17 | 2 | 1.37% | 26.1% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.31% | -17.2% |
| retest2 (combined) | 19 | 6 | 31.6% | 0 | 17 | 2 | 1.37% | 26.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 11:15:00 | 1633.10 | 1587.58 | 1587.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 14:15:00 | 1640.30 | 1588.99 | 1588.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 1850.75 | 1860.71 | 1781.56 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-09 10:15:00 | 1877.30 | 1860.31 | 1783.31 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-09 11:15:00 | 1862.00 | 1860.33 | 1783.70 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-10-09 12:15:00 | 1871.70 | 1860.44 | 1784.14 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 13:15:00 | 1877.35 | 1860.61 | 1784.61 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 09:15:00 | 1879.55 | 1860.91 | 1785.89 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 10:15:00 | 1881.05 | 1861.11 | 1786.36 | BUY ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-10 12:15:00 | 1871.10 | 1861.27 | 1787.19 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 13:15:00 | 1879.55 | 1861.46 | 1787.65 | BUY ENTRY1 attempt 3/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2024-10-11 10:15:00 | 1875.35 | 1861.92 | 1789.35 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1876.80 | 1862.07 | 1789.78 | BUY ENTRY1 attempt 4/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1811.00 | 1859.50 | 1799.53 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-10-18 11:15:00 | 1825.65 | 1858.79 | 1799.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:15:00 | 1823.40 | 1858.44 | 1799.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.75 | 1800.20 | SL hit (close<ema400) qty=1.00 sl=1800.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.75 | 1800.20 | SL hit (close<ema400) qty=1.00 sl=1800.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.75 | 1800.20 | SL hit (close<ema400) qty=1.00 sl=1800.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1797.80 | 1856.75 | 1800.20 | SL hit (close<ema400) qty=1.00 sl=1800.20 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 1775.35 | 1855.33 | 1800.04 | SL hit (close<static) qty=1.00 sl=1793.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 10:15:00 | 1738.50 | 1679.28 | 1679.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-28 11:15:00 | 1749.05 | 1685.54 | 1682.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.16 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1693.40 | 1704.55 | 1693.16 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-01 13:15:00 | 1751.55 | 1705.01 | 1693.45 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1750.00 | 1705.46 | 1693.74 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-12 10:15:00 | 1760.85 | 1737.71 | 1714.78 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-12 11:15:00 | 1774.00 | 1738.07 | 1715.07 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-03-28 10:15:00 | 2012.50 | 1858.01 | 1814.36 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1853.40 | 1880.96 | 1833.85 | SL hit (close<ema200) qty=0.50 sl=1880.96 alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-21 09:15:00 | 2040.10 | 1905.60 | 1857.15 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1937.90 | 1969.77 | 1903.98 | SL hit (close<ema200) qty=0.50 sl=1969.77 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 11:15:00 | 2036.30 | 1978.32 | 1978.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 2038.20 | 1978.92 | 1978.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 2015.00 | 2025.48 | 2006.19 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 11:15:00 | 2001.40 | 2025.09 | 2006.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 2001.40 | 2025.09 | 2006.19 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-09-29 14:15:00 | 2025.10 | 2023.28 | 2006.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 15:15:00 | 2012.00 | 2023.17 | 2006.21 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-01 09:15:00 | 1983.50 | 2021.89 | 2006.22 | SL hit (close<static) qty=1.00 sl=2000.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-06 09:15:00 | 2025.00 | 2018.75 | 2005.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 2021.80 | 2018.78 | 2005.74 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-08 12:15:00 | 2016.50 | 2020.72 | 2007.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 2014.30 | 2020.65 | 2007.79 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-09 10:15:00 | 2020.20 | 2020.33 | 2007.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 11:15:00 | 2022.00 | 2020.35 | 2007.95 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 2010.00 | 2020.03 | 2008.27 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-13 14:15:00 | 2019.40 | 2019.19 | 2008.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 15:15:00 | 2023.90 | 2019.23 | 2008.44 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-14 14:15:00 | 2019.00 | 2019.20 | 2008.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 15:15:00 | 2019.90 | 2019.21 | 2008.80 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=2000.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=2000.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=2000.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=2005.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1978.70 | 2079.64 | 2053.12 | SL hit (close<static) qty=1.00 sl=2005.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-12 12:15:00 | 2028.50 | 2071.35 | 2050.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 13:15:00 | 2034.90 | 2070.99 | 2050.08 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-12-17 13:15:00 | 2017.30 | 2065.77 | 2058.93 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 2019.80 | 2065.31 | 2058.73 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 2020.50 | 2064.87 | 2058.54 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-18 10:15:00 | 2033.60 | 2064.14 | 2058.24 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:15:00 | 2034.90 | 2063.85 | 2058.12 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 2014.60 | 2056.58 | 2055.12 | SL hit (close<static) qty=1.00 sl=2018.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 2003.50 | 2054.46 | 2054.09 | SL hit (close<static) qty=1.00 sl=2005.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 2003.50 | 2054.46 | 2054.09 | SL hit (close<static) qty=1.00 sl=2005.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-30 14:15:00 | 2026.50 | 2049.12 | 2051.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 15:15:00 | 2040.20 | 2049.03 | 2051.31 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 2014.10 | 2048.69 | 2051.12 | SL hit (close<static) qty=1.00 sl=2018.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-12-31 11:15:00 | 2029.70 | 2048.20 | 2050.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 12:15:00 | 2033.20 | 2048.05 | 2050.77 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 2014.50 | 2045.03 | 2048.64 | SL hit (close<static) qty=1.00 sl=2018.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-04 13:15:00 | 2030.00 | 1989.69 | 2011.71 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-04 14:15:00 | 2021.80 | 1990.01 | 2011.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-06 14:15:00 | 2024.30 | 1991.94 | 2011.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 15:15:00 | 2020.80 | 1992.23 | 2011.33 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-09 09:15:00 | 2028.80 | 1992.59 | 2011.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 2024.10 | 1992.91 | 2011.48 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2025.00 | 1999.08 | 2012.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-02-12 10:15:00 | 2033.00 | 1999.41 | 2013.05 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 2033.00 | 1999.75 | 2013.15 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-13 12:15:00 | 2017.10 | 2001.97 | 2013.76 | SL hit (close<static) qty=1.00 sl=2018.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-16 09:15:00 | 2039.90 | 2002.94 | 2014.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:15:00 | 2041.90 | 2003.33 | 2014.16 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2049.70 | 2022.50 | 2022.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 2052.80 | 2023.00 | 2022.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 2016.60 | 2024.11 | 2023.29 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-10-09 13:15:00 | 1877.35 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.24% |
| BUY | retest1 | 2024-10-10 10:15:00 | 1881.05 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.43% |
| BUY | retest1 | 2024-10-10 13:15:00 | 1879.55 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest1 | 2024-10-11 11:15:00 | 1876.80 | 2024-10-21 09:15:00 | 1797.80 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-10-18 12:15:00 | 1823.40 | 2024-10-21 11:15:00 | 1775.35 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1750.00 | 2025-03-28 10:15:00 | 2012.50 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1750.00 | 2025-04-07 09:15:00 | 1853.40 | STOP_HIT | 0.50 | 5.91% |
| BUY | retest2 | 2025-02-12 11:15:00 | 1774.00 | 2025-04-21 09:15:00 | 2040.10 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-02-12 11:15:00 | 1774.00 | 2025-04-30 09:15:00 | 1937.90 | STOP_HIT | 0.50 | 9.24% |
| BUY | retest2 | 2025-09-29 15:15:00 | 2012.00 | 2025-10-01 09:15:00 | 1983.50 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-10-06 10:15:00 | 2021.80 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-08 13:15:00 | 2014.30 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-10-09 11:15:00 | 2022.00 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-10-13 15:15:00 | 2023.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-10-14 15:15:00 | 2019.90 | 2025-11-11 09:15:00 | 1978.70 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-11-12 13:15:00 | 2034.90 | 2025-12-26 12:15:00 | 2014.60 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-12-17 14:15:00 | 2019.80 | 2025-12-29 10:15:00 | 2003.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-18 11:15:00 | 2034.90 | 2025-12-29 10:15:00 | 2003.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-30 15:15:00 | 2040.20 | 2025-12-31 09:15:00 | 2014.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-12-31 12:15:00 | 2033.20 | 2026-01-08 09:15:00 | 2014.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-02-09 10:15:00 | 2024.10 | 2026-02-13 12:15:00 | 2017.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2026-02-12 11:15:00 | 2033.00 | 2026-02-25 14:15:00 | 2049.70 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2026-02-16 10:15:00 | 2041.90 | 2026-02-25 14:15:00 | 2049.70 | STOP_HIT | 1.00 | 0.38% |
