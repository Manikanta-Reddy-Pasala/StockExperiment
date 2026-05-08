# ADANIENT (ADANIENT)

## Backtest Summary

- **Window:** 2025-08-14 09:15:00 → 2026-05-08 15:15:00 (1237 bars)
- **Last close:** 2505.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty @ 15% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 15%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 0 |
| ALERT3 | 2 |
| PENDING | 7 |
| PENDING_CANCEL | 1 |
| ENTRY1 | 2 |
| ENTRY2 | 4 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 2
- **Target hits / Stop hits / Partials:** 0 / 6 / 4
- **Avg / median % per leg:** 6.92% / 7.03%
- **Sum % (uncompounded):** 69.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 8 | 80.0% | 0 | 6 | 4 | 6.92% | 69.2% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 8.94% | 35.8% |
| SELL @ 3rd Alert (retest2) | 6 | 4 | 66.7% | 0 | 4 | 2 | 5.58% | 33.5% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 8.94% | 35.8% |
| retest2 (combined) | 6 | 4 | 66.7% | 0 | 4 | 2 | 5.58% | 33.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 14:15:00 | 2256.50 | 2436.02 | 2436.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 2249.00 | 2434.16 | 2435.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2279.60 | 2277.03 | 2324.48 | EMA200 retest candle locked (from downside) |
| Cross detected — sustain check pending | 2026-01-06 10:15:00 | 2265.00 | 2277.43 | 2322.14 | ENTRY1 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-01-06 11:15:00 | 2271.20 | 2277.36 | 2321.89 | ENTRY1 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-06 13:15:00 | 2254.80 | 2277.08 | 2321.30 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 14:15:00 | 2258.90 | 2276.90 | 2320.99 | SELL ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Cross detected — sustain check pending | 2026-01-08 09:15:00 | 2253.10 | 2276.24 | 2318.72 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:15:00 | 2234.90 | 2275.83 | 2318.30 | SELL ENTRY1 attempt 2/4 (retest1 break sustained 60m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 13:15:00 | 1920.06 | 2192.85 | 2258.03 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 13:15:00 | 1899.66 | 2192.85 | 2258.03 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2182.00 | 2128.57 | 2211.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2182.00 | 2128.57 | 2211.30 | SL hit (close>ema200) qty=0.50 sl=2128.57 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2182.00 | 2128.57 | 2211.30 | SL hit (close>ema200) qty=0.50 sl=2128.57 alert=retest1 |
| Cross detected — sustain check pending | 2026-02-13 09:15:00 | 2154.50 | 2169.58 | 2214.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 10:15:00 | 2157.10 | 2169.46 | 2214.21 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 2234.20 | 2169.63 | 2210.81 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |
| Cross detected — sustain check pending | 2026-02-19 14:15:00 | 2155.70 | 2173.82 | 2209.87 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 2156.70 | 2173.65 | 2209.60 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-02-27 15:15:00 | 2157.00 | 2179.31 | 2205.81 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 2093.00 | 2178.45 | 2205.24 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 3960m) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 11:15:00 | 1833.19 | 2056.60 | 2121.40 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1779.05 | 1998.11 | 2080.70 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2005.00 | 1961.09 | 2049.12 | SL hit (close>ema200) qty=0.50 sl=1961.09 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 2005.00 | 1961.09 | 2049.12 | SL hit (close>ema200) qty=0.50 sl=1961.09 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-15 13:15:00 | 2152.80 | 1997.51 | 2055.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 2144.00 | 1998.96 | 2055.89 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 15:15:00 | 2139.90 | 2000.37 | 2056.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-20 14:15:00 | 2233.80 | 2030.07 | 2067.42 | SL hit (close>static) qty=1.00 sl=2232.10 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 2316.10 | 2099.13 | 2098.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2102.52 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-01-23 13:15:00 | 1920.06 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-01-23 13:15:00 | 1899.66 | PARTIAL | 0.50 | 15.00% |
| SELL | retest1 | 2026-01-06 14:15:00 | 2258.90 | 2026-02-03 09:15:00 | 2182.00 | STOP_HIT | 0.50 | 3.40% |
| SELL | retest1 | 2026-01-08 10:15:00 | 2234.90 | 2026-02-03 09:15:00 | 2182.00 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2026-02-13 10:15:00 | 2157.10 | 2026-02-17 12:15:00 | 2234.20 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-03-23 11:15:00 | 1833.19 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-03-30 14:15:00 | 1779.05 | PARTIAL | 0.50 | 15.00% |
| SELL | retest2 | 2026-02-19 15:15:00 | 2156.70 | 2026-04-08 09:15:00 | 2005.00 | STOP_HIT | 0.50 | 7.03% |
| SELL | retest2 | 2026-03-02 09:15:00 | 2093.00 | 2026-04-08 09:15:00 | 2005.00 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2026-04-15 14:15:00 | 2144.00 | 2026-04-20 14:15:00 | 2233.80 | STOP_HIT | 1.00 | -4.19% |
