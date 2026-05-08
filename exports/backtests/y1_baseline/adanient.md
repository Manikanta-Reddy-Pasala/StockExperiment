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
| CROSSOVER | 6 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 3 |
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

### Cycle 1 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 2238.93 | 2441.83 | 2442.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 2196.08 | 2431.42 | 2436.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2308.54 | 2272.56 | 2323.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 10:15:00 | 2333.36 | 2273.17 | 2323.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2333.36 | 2273.17 | 2323.37 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 09:15:00 | 2544.90 | 2352.54 | 2352.32 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 15:15:00 | 2297.87 | 2394.95 | 2394.98 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2436.22 | 2395.36 | 2395.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 2462.78 | 2398.33 | 2396.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 2412.30 | 2413.00 | 2405.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 2402.00 | 2412.96 | 2405.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 2402.00 | 2412.96 | 2405.34 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-11-27 13:15:00)

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

### Cycle 6 — BUY (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 15:15:00 | 2281.60 | 2094.05 | 2093.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 2309.40 | 2096.19 | 2095.06 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-08 11:15:00 | 2220.70 | 2026-01-20 09:15:00 | 2109.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-01-08 11:15:00 | 2220.70 | 2026-01-23 12:15:00 | 1998.63 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-16 09:15:00 | 2130.00 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -4.89% |
| SELL | retest2 | 2026-03-02 10:15:00 | 2112.40 | 2026-03-09 09:15:00 | 2006.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 10:15:00 | 2112.40 | 2026-03-23 09:15:00 | 1901.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-20 09:15:00 | 2235.30 | STOP_HIT | 1.00 | -4.93% |
