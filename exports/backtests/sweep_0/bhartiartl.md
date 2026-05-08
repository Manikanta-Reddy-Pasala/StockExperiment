# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 1834.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 1 |
| ALERT1 | 1 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 9 |
| PENDING_CANCEL | 2 |
| ENTRY1 | 1 |
| ENTRY2 | 6 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 4
- **Avg / median % per leg:** 5.59% / 5.49%
- **Sum % (uncompounded):** 55.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 5.59% | 55.9% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.37% | -2.4% |
| SELL @ 3rd Alert (retest2) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.48% | 58.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.37% | -2.4% |
| retest2 (combined) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.48% | 58.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2002.80 | 2056.72 | 2056.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 1997.40 | 2055.12 | 2055.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 10:15:00 | 2022.20 | 2019.76 | 2035.60 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 2008.90 | 2019.88 | 2035.19 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-05 10:15:00 | 2020.10 | 2019.88 | 2035.12 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-05 12:15:00 | 1999.70 | 2019.59 | 2034.82 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 13:15:00 | 1987.10 | 2019.27 | 2034.58 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 2034.10 | 2018.71 | 2033.92 | SL hit (close>ema400) qty=1.00 sl=2033.92 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-10 13:15:00 | 2011.20 | 2021.03 | 2033.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 2012.10 | 2020.94 | 2033.85 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 2019.90 | 2020.25 | 2032.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-12 12:15:00 | 2020.60 | 2020.25 | 2032.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-12 14:15:00 | 2014.50 | 2020.21 | 2032.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 2016.20 | 2020.17 | 2032.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-13 10:15:00 | 2010.40 | 2020.11 | 2032.36 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2004.80 | 2019.96 | 2032.22 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 2015.30 | 2019.84 | 2030.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 2015.30 | 2019.79 | 2030.57 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1911.49 | 2005.87 | 2021.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1915.39 | 2005.87 | 2021.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1914.53 | 2005.87 | 2021.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 14:15:00 | 1904.56 | 2004.97 | 2021.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1810.89 | 1948.23 | 1984.82 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1814.58 | 1948.23 | 1984.82 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1804.32 | 1948.23 | 1984.82 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1813.77 | 1948.23 | 1984.82 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1891.90 | 1847.74 | 1883.85 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 1869.80 | 1849.85 | 1883.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 1871.60 | 1850.07 | 1883.79 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 1903.70 | 1851.17 | 1883.84 | SL hit (close>static) qty=1.00 sl=1900.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-04 09:15:00 | 1866.00 | 1852.06 | 1883.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 1856.80 | 1852.11 | 1883.67 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-05 13:15:00 | 1987.10 | 2026-02-06 11:15:00 | 2034.10 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2026-02-10 14:15:00 | 2012.10 | 2026-02-25 13:15:00 | 1911.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 2016.20 | 2026-02-25 13:15:00 | 1915.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2004.80 | 2026-02-25 13:15:00 | 1914.53 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2026-02-19 10:15:00 | 2015.30 | 2026-02-25 14:15:00 | 1904.56 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2026-02-10 14:15:00 | 2012.10 | 2026-03-11 11:15:00 | 1810.89 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 2016.20 | 2026-03-11 11:15:00 | 1814.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2004.80 | 2026-03-11 11:15:00 | 1804.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 2015.30 | 2026-03-11 11:15:00 | 1813.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1871.60 | 2026-04-30 13:15:00 | 1903.70 | STOP_HIT | 1.00 | -1.72% |
