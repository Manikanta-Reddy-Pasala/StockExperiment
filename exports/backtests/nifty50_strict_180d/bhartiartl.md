# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1834.50
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
| PENDING | 8 |
| PENDING_CANCEL | 1 |
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
- **Avg / median % per leg:** 5.76% / 5.49%
- **Sum % (uncompounded):** 57.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 5.76% | 57.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.69% | -0.7% |
| SELL @ 3rd Alert (retest2) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.48% | 58.3% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.69% | -0.7% |
| retest2 (combined) | 9 | 8 | 88.9% | 4 | 1 | 4 | 6.48% | 58.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 13:15:00 | 1966.00 | 2049.33 | 2049.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 1964.00 | 2047.01 | 2048.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 12:15:00 | 2024.20 | 2024.04 | 2035.48 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-05 09:15:00 | 2009.30 | 2023.96 | 2035.21 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-05 10:15:00 | 2020.10 | 2023.92 | 2035.13 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 2034.10 | 2022.43 | 2033.93 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-06 11:15:00 | 2034.10 | 2022.43 | 2033.93 | SL hit (close>ema400) qty=1.00 sl=2033.93 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-10 13:15:00 | 2011.10 | 2024.19 | 2033.97 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 14:15:00 | 2012.20 | 2024.07 | 2033.86 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-12 11:15:00 | 2019.90 | 2023.04 | 2032.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-12 12:15:00 | 2020.70 | 2023.02 | 2032.74 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-02-12 14:15:00 | 2014.50 | 2022.92 | 2032.60 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 15:15:00 | 2014.50 | 2022.84 | 2032.51 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-13 10:15:00 | 2010.40 | 2022.72 | 2032.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 11:15:00 | 2004.80 | 2022.54 | 2032.22 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-19 09:15:00 | 2015.30 | 2021.86 | 2030.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 10:15:00 | 2015.30 | 2021.79 | 2030.58 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1911.59 | 2007.35 | 2021.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1913.77 | 2007.35 | 2021.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 13:15:00 | 1914.54 | 2007.35 | 2021.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 14:15:00 | 1904.56 | 2006.43 | 2021.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1810.98 | 1948.96 | 1984.78 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1813.05 | 1948.96 | 1984.78 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1804.32 | 1948.96 | 1984.78 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-11 11:15:00 | 1813.77 | 1948.96 | 1984.78 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 1892.00 | 1847.85 | 1884.52 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-30 09:15:00 | 1869.80 | 1849.95 | 1884.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 10:15:00 | 1871.60 | 1850.17 | 1884.43 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-30 13:15:00 | 1903.70 | 1851.27 | 1884.48 | SL hit (close>static) qty=1.00 sl=1900.20 alert=retest2 |
| Cross detected — sustain check pending | 2026-05-04 09:15:00 | 1866.00 | 1852.11 | 1884.41 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:15:00 | 1857.00 | 1852.16 | 1884.27 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-02-05 10:15:00 | 2020.10 | 2026-02-06 11:15:00 | 2034.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-02-10 14:15:00 | 2012.20 | 2026-02-25 13:15:00 | 1911.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 2014.50 | 2026-02-25 13:15:00 | 1913.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2004.80 | 2026-02-25 13:15:00 | 1914.54 | PARTIAL | 0.50 | 4.50% |
| SELL | retest2 | 2026-02-19 10:15:00 | 2015.30 | 2026-02-25 14:15:00 | 1904.56 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2026-02-10 14:15:00 | 2012.20 | 2026-03-11 11:15:00 | 1810.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-12 15:15:00 | 2014.50 | 2026-03-11 11:15:00 | 1813.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 11:15:00 | 2004.80 | 2026-03-11 11:15:00 | 1804.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-19 10:15:00 | 2015.30 | 2026-03-11 11:15:00 | 1813.77 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-30 10:15:00 | 1871.60 | 2026-04-30 13:15:00 | 1903.70 | STOP_HIT | 1.00 | -1.72% |
