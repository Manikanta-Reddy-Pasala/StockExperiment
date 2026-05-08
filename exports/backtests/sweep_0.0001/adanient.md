# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2025-05-09 09:15:00 → 2026-05-08 15:15:00 (1731 bars)
- **Last close:** 2502.00
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
| ALERT3 | 1 |
| PENDING | 4 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 3 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 3.36% / 5.00%
- **Sum % (uncompounded):** 20.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 3.36% | 20.2% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| SELL @ 3rd Alert (retest2) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.29% | 5.2% |
| retest1 (combined) | 2 | 2 | 100.0% | 1 | 0 | 1 | 7.50% | 15.0% |
| retest2 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.29% | 5.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 2279.00 | 2397.85 | 2398.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 2256.30 | 2396.44 | 2397.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 2272.00 | 2270.61 | 2310.59 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-08 10:15:00 | 2234.90 | 2270.37 | 2303.88 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:15:00 | 2220.70 | 2269.87 | 2303.46 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 2109.66 | 2228.95 | 2273.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-23 12:15:00 | 1998.63 | 2193.01 | 2249.69 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2181.90 | 2117.03 | 2195.41 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-13 15:15:00 | 2127.00 | 2161.91 | 2200.77 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2130.00 | 2161.60 | 2200.42 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 3960m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2163.97 | 2199.72 | SL hit (close>static) qty=1.00 sl=2233.40 alert=retest2 |
| Cross detected — sustain check pending | 2026-03-02 09:15:00 | 2093.00 | 2175.42 | 2197.07 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 10:15:00 | 2112.40 | 2174.79 | 2196.65 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 2006.78 | 2149.27 | 2180.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 1901.16 | 2059.85 | 2119.22 | Target hit (10%) qty=0.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-13 14:15:00 | 2124.00 | 1986.82 | 2048.95 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2130.20 | 1988.25 | 2049.36 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 2235.30 | 2027.84 | 2063.81 | SL hit (close>static) qty=1.00 sl=2233.40 alert=retest2 |
| CROSSOVER_SKIP | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.98 | min_gap filter: gap=0.003% < 0.010% |
| TREND_RESET | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.98 | EMA inversion without crossover edge (EMA200=2094.05 EMA400=2093.98) — end cycle |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-08 11:15:00 | 2220.70 | 2026-01-20 09:15:00 | 2109.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-08 11:15:00 | 2220.70 | 2026-01-23 12:15:00 | 1998.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-16 09:15:00 | 2130.00 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-03-02 10:15:00 | 2112.40 | 2026-03-09 09:15:00 | 2006.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:15:00 | 2112.40 | 2026-03-23 09:15:00 | 1901.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-20 09:15:00 | 2235.30 | STOP_HIT | 1.00 | -4.93% |
