# Balkrishna Industries Ltd. (BALKRISIND)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 2265.10
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
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 4 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 0 / 5 / 1
- **Avg / median % per leg:** -1.05% / -1.67%
- **Sum % (uncompounded):** -6.32%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.05% | -6.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.10% | -0.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -5.82% | -5.8% |
| retest2 (combined) | 5 | 2 | 40.0% | 0 | 4 | 1 | -0.10% | -0.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 09:15:00 | 2239.00 | 2423.36 | 2424.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 2233.50 | 2411.85 | 2418.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2221.40 | 2209.58 | 2287.33 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:45:00 | 2187.80 | 2209.99 | 2284.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2277.20 | 2211.94 | 2282.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:30:00 | 2277.90 | 2211.94 | 2282.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 2281.80 | 2212.64 | 2282.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 12:00:00 | 2281.80 | 2212.64 | 2282.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 2271.10 | 2213.22 | 2282.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:00:00 | 2258.10 | 2213.66 | 2282.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 2262.00 | 2214.31 | 2282.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>ema400) qty=1.00 sl=2280.67 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 2315.20 | 2219.34 | 2280.67 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 10:15:00 | 2264.20 | 2238.77 | 2284.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 12:15:00 | 2302.00 | 2240.17 | 2284.34 | SL hit (close>static) qty=1.00 sl=2285.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 12:30:00 | 2259.90 | 2247.68 | 2285.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 2146.91 | 2238.53 | 2274.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-06 15:15:00 | 2236.00 | 2228.41 | 2264.84 | SL hit (close>ema200) qty=0.50 sl=2228.41 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 2269.70 | 2229.25 | 2264.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 12:45:00 | 2257.70 | 2229.25 | 2264.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 2267.90 | 2229.63 | 2264.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:30:00 | 2269.00 | 2229.63 | 2264.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 2271.00 | 2230.28 | 2264.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 2266.90 | 2230.28 | 2264.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2285.60 | 2230.83 | 2264.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 2285.60 | 2230.83 | 2264.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 2266.00 | 2231.18 | 2264.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:45:00 | 2297.80 | 2231.18 | 2264.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 2265.10 | 2233.61 | 2265.05 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-04-09 09:45:00 | 2187.80 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -5.82% |
| SELL | retest2 | 2026-04-10 14:00:00 | 2258.10 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-10 15:15:00 | 2262.00 | 2026-04-15 14:15:00 | 2315.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-04-21 10:15:00 | 2264.20 | 2026-04-21 12:15:00 | 2302.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-04-30 10:15:00 | 2146.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-23 12:30:00 | 2259.90 | 2026-05-06 15:15:00 | 2236.00 | STOP_HIT | 0.50 | 1.06% |
