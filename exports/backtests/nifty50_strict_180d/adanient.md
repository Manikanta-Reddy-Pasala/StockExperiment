# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 2505.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 1 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 4 / 2 / 4
- **Avg / median % per leg:** 5.22% / 5.00%
- **Sum % (uncompounded):** 52.24%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 4 | 2 | 4 | 5.22% | 52.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 2 | 2 | 2 | 3.71% | 22.2% |
| retest1 (combined) | 4 | 4 | 100.0% | 2 | 0 | 2 | 7.50% | 30.0% |
| retest2 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 3.71% | 22.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.50 | 2434.23 | 2435.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2432.38 | 2434.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2279.60 | 2276.71 | 2323.89 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 2265.00 | 2277.14 | 2321.59 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 2271.20 | 2277.09 | 2321.34 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-06 13:15:00 | 2254.80 | 2276.81 | 2320.76 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2258.90 | 2276.63 | 2320.45 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 2253.10 | 2275.99 | 2318.20 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 2234.90 | 2275.58 | 2317.78 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 14:15:00 | 2145.95 | 2266.71 | 2310.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2123.15 | 2264.46 | 2309.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-01-21 10:15:00 | 2033.01 | 2219.72 | 2276.32 | Target hit (10%) qty=0.50 alert=retest1 |
| Target hit | 2026-01-21 10:15:00 | 2011.41 | 2219.72 | 2276.32 | Target hit (10%) qty=0.50 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2182.00 | 2128.49 | 2211.01 | EMA400 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2154.50 | 2169.54 | 2214.27 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 2157.10 | 2169.41 | 2213.99 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2169.59 | 2210.61 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-19 14:15:00 | 2155.70 | 2173.79 | 2209.68 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 2156.70 | 2173.62 | 2209.42 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 15:15:00 | 2157.00 | 2179.29 | 2205.65 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2093.00 | 2178.43 | 2205.09 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 09:15:00 | 2048.86 | 2173.32 | 2201.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1988.35 | 2151.63 | 2187.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-19 12:15:00 | 1941.03 | 2076.01 | 2135.01 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-23 09:15:00 | 1883.70 | 2061.05 | 2124.18 | Target hit (10%) qty=0.50 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 13:15:00 | 2152.80 | 1997.50 | 2055.39 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 2144.00 | 1998.96 | 2055.83 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2233.80 | 2030.07 | 2067.37 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 2316.10 | 2099.13 | 2098.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.44 | 2102.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-09 14:15:00 | 2145.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-12 09:15:00 | 2123.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-21 10:15:00 | 2033.01 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-21 10:15:00 | 2011.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-13 10:15:00 | 2157.10 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-03-04 09:15:00 | 2048.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-03-09 09:15:00 | 1988.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-03-19 12:15:00 | 1941.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-03-23 09:15:00 | 1883.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-15 14:15:00 | 2144.00 | 2026-04-20 14:15:00 | 2233.80 | STOP_HIT | 1.00 | -4.19% |
