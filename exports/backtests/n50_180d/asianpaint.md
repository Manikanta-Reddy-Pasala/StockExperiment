# ASIANPAINT (ASIANPAINT)

## Backtest Summary

- **Window:** 2024-10-07 09:15:00 → 2026-05-08 15:15:00 (2741 bars)
- **Last close:** 2600.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 5 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / Stop hits / Partials:** 0 / 9 / 0
- **Avg / median % per leg:** -3.42% / -3.89%
- **Sum % (uncompounded):** -30.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.70% | -3.4% |
| SELL (all) | 7 | 0 | 0.0% | 0 | 7 | 0 | -3.91% | -27.4% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.14% | -16.6% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -3.60% | -10.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.14% | -16.6% |
| retest2 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -2.84% | -14.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.80 | 2720.57 | 2721.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2374.00 | 2700.46 | 2710.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2263.38 | 2367.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:15:00 | 2264.60 | 2263.38 | 2367.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:00:00 | 2269.10 | 2263.44 | 2366.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 14:00:00 | 2276.00 | 2263.52 | 2365.24 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 14:30:00 | 2272.50 | 2263.73 | 2364.84 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 2364.60 | 2265.40 | 2361.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2357.60 | 2266.31 | 2361.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 09:15:00 | 2300.60 | 2271.00 | 2361.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 11:30:00 | 2344.70 | 2273.08 | 2360.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2348.00 | 2275.46 | 2360.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 2414.90 | 2277.56 | 2361.05 | SL hit (close>static) qty=1.00 sl=2372.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 2414.90 | 2277.56 | 2361.05 | SL hit (close>static) qty=1.00 sl=2372.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 2414.90 | 2277.56 | 2361.05 | SL hit (close>static) qty=1.00 sl=2372.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 2552.30 | 2410.96 | 2410.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 2557.00 | 2415.95 | 2413.37 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2026-01-19 10:45:00 | 2778.50 | 2026-01-19 15:15:00 | 2736.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-23 11:00:00 | 2782.00 | 2026-01-23 13:15:00 | 2729.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest1 | 2026-04-08 10:15:00 | 2264.60 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest1 | 2026-04-08 11:00:00 | 2269.10 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest1 | 2026-04-08 14:00:00 | 2276.00 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest1 | 2026-04-08 14:30:00 | 2272.50 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2300.60 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2026-04-13 11:30:00 | 2344.70 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-04-13 15:15:00 | 2348.00 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -2.85% |
