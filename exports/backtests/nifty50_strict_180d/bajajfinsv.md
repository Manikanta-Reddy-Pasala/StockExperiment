# BAJAJFINSV (BAJAJFINSV)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:30:00 (1238 bars)
- **Last close:** 1818.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 2 |
| PENDING | 5 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 2 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 5 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / Stop hits / Partials:** 0 / 4 / 1
- **Avg / median % per leg:** -0.87% / -0.79%
- **Sum % (uncompounded):** -4.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.87% | -4.3% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.00% | -10.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.89% | 5.7% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.00% | -10.0% |
| retest2 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 1.89% | 5.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2035.50 | 2045.79 | 2045.80 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 2073.20 | 2045.93 | 2045.87 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 11:15:00 | 2032.90 | 2045.71 | 2045.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 12:15:00 | 2030.00 | 2045.55 | 2045.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2010.50 | 1990.79 | 2012.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 2010.50 | 1990.79 | 2012.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2010.50 | 1990.79 | 2012.53 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-06 10:15:00 | 1987.90 | 1994.55 | 2012.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-02-06 11:15:00 | 2007.50 | 1994.67 | 2012.24 | ENTRY2 sustain failed after 60m |

### Cycle 4 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 2066.60 | 2022.66 | 2022.50 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1946.00 | 2022.20 | 2022.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1941.50 | 2021.39 | 2021.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1805.00 | 1783.53 | 1866.13 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-09 12:15:00 | 1775.00 | 1783.40 | 1862.44 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 13:15:00 | 1763.00 | 1783.20 | 1861.94 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-13 09:15:00 | 1769.60 | 1784.33 | 1858.68 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1774.40 | 1784.23 | 1858.26 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1849.40 | 1796.77 | 1853.37 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-04-22 09:15:00 | 1840.80 | 1798.27 | 1853.29 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1842.70 | 1798.71 | 1853.23 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.24 | SL hit (close>ema400) qty=1.00 sl=1853.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.24 | SL hit (close>ema400) qty=1.00 sl=1853.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 1857.20 | 1799.80 | 1853.24 | SL hit (close>static) qty=1.00 sl=1854.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-22 15:15:00 | 1842.90 | 1801.21 | 1853.15 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1818.00 | 1801.37 | 1852.97 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 1080m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 14:15:00 | 1727.10 | 1792.37 | 1838.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1791.30 | 1789.31 | 1834.48 | SL hit (close>ema200) qty=0.50 sl=1789.31 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-09 13:15:00 | 1763.00 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest1 | 2026-04-13 10:15:00 | 1774.40 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1842.70 | 2026-04-22 12:15:00 | 1857.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1818.00 | 2026-04-30 14:15:00 | 1727.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 09:15:00 | 1818.00 | 2026-05-05 12:15:00 | 1791.30 | STOP_HIT | 0.50 | 1.47% |
