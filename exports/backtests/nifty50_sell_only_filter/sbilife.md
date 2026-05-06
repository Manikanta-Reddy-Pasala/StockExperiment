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
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 4 |
| PENDING | 14 |
| PENDING_CANCEL | 5 |
| ENTRY1 | 3 |
| ENTRY2 | 6 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 11 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 4
- **Target hits / Stop hits / Partials:** 0 / 9 / 2
- **Avg / median % per leg:** 6.81% / 1.78%
- **Sum % (uncompounded):** 74.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 11 | 7 | 63.6% | 0 | 9 | 2 | 6.81% | 74.9% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.70% | 5.1% |
| BUY @ 3rd Alert (retest2) | 8 | 4 | 50.0% | 0 | 6 | 2 | 8.73% | 69.8% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 3 | 3 | 100.0% | 0 | 3 | 0 | 1.70% | 5.1% |
| retest2 (combined) | 8 | 4 | 50.0% | 0 | 6 | 2 | 8.73% | 69.8% |

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

### Cycle 2 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 1543.25 | 1473.88 | 1473.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1551.00 | 1475.90 | 1474.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.95 | 1496.19 | 1486.14 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1464.95 | 1496.19 | 1486.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1464.95 | 1496.19 | 1486.14 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 1487.90 | 1492.62 | 1484.88 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 15:15:00 | 1488.00 | 1492.57 | 1484.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-11 09:15:00 | 1523.00 | 1492.45 | 1485.13 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 1523.90 | 1492.76 | 1485.32 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-25 09:15:00 | 1711.20 | 1536.81 | 1511.61 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-04-25 09:15:00 | 1752.49 | 1536.81 | 1511.61 | Partial book 0.50 @ 15%; trail SL->entry alert=retest2 |
| CROSSOVER_SKIP | 2025-10-07 15:15:00 | 1784.10 | 1809.06 | 1809.16 | HTF filter: close above htf_sma |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 1850.00 | 1808.95 | 1808.93 | Force close (CROSSOVER_FLIP) qty=0.50 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-15 11:15:00)

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
| BUY | retest2 | 2025-04-08 15:15:00 | 1488.00 | 2025-04-25 09:15:00 | 1711.20 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-11 10:15:00 | 1523.90 | 2025-04-25 09:15:00 | 1752.49 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-04-08 15:15:00 | 1488.00 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 0.50 | 24.33% |
| BUY | retest2 | 2025-04-11 10:15:00 | 1523.90 | 2025-10-15 11:15:00 | 1850.00 | STOP_HIT | 0.50 | 21.40% |
| BUY | retest1 | 2025-12-02 14:15:00 | 1980.30 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.49% |
| BUY | retest1 | 2025-12-03 12:15:00 | 1974.70 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest1 | 2025-12-03 15:15:00 | 1974.00 | 2026-01-22 14:15:00 | 2009.85 | STOP_HIT | 1.00 | 1.82% |
| BUY | retest2 | 2026-01-27 12:15:00 | 2034.00 | 2026-01-29 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-03 10:15:00 | 2050.30 | 2026-02-03 13:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-02-04 13:15:00 | 2042.50 | 2026-02-05 12:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2026-02-13 14:15:00 | 2032.60 | 2026-03-02 09:15:00 | 2009.70 | STOP_HIT | 1.00 | -1.13% |
