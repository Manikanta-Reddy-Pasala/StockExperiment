# Asian Paints Ltd. (ASIANPAINT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 12 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 13
- **Target hits / Stop hits / Partials:** 2 / 14 / 1
- **Avg / median % per leg:** -0.64% / -1.91%
- **Sum % (uncompounded):** -10.95%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 2 | 33.3% | 2 | 4 | 0 | 2.04% | 12.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 2 | 4 | 0 | 2.04% | 12.3% |
| SELL (all) | 11 | 2 | 18.2% | 0 | 10 | 1 | -2.11% | -23.2% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.14% | -16.6% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.95% | -6.6% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.14% | -16.6% |
| retest2 (combined) | 13 | 4 | 30.8% | 2 | 10 | 1 | 0.43% | 5.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 2307.90 | 2347.65 | 2347.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 2343.80 | 2346.80 | 2347.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 2341.80 | 2346.75 | 2347.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 14:00:00 | 2329.60 | 2346.58 | 2347.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 09:15:00 | 2213.12 | 2299.65 | 2320.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 2280.60 | 2280.02 | 2305.43 | SL hit (close>ema200) qty=0.50 sl=2280.02 alert=retest2 |

### Cycle 2 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2428.00 | 2316.53 | 2316.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.84 | 2316.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.85 | 2350.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 13:00:00 | 2369.70 | 2371.85 | 2350.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.73 | 2351.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:00:00 | 2344.00 | 2370.73 | 2351.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 2347.70 | 2370.50 | 2351.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:00:00 | 2347.70 | 2370.50 | 2351.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2350.20 | 2370.30 | 2351.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 2345.20 | 2370.30 | 2351.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 2346.00 | 2369.89 | 2351.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 2341.00 | 2369.89 | 2351.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 2336.60 | 2369.56 | 2351.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 2343.10 | 2369.56 | 2351.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 2348.30 | 2367.06 | 2351.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 2348.30 | 2367.06 | 2351.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 2344.90 | 2366.84 | 2351.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:30:00 | 2353.00 | 2366.80 | 2351.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 13:00:00 | 2350.50 | 2365.97 | 2351.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 14:00:00 | 2398.10 | 2366.29 | 2351.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 10:15:00 | 2588.30 | 2430.54 | 2394.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.74 | 2447.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.28 | 2441.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2422.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-16 14:30:00 | 2417.90 | 2404.32 | 2422.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2514.70 | 2405.47 | 2422.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 2514.70 | 2405.47 | 2422.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2522.10 | 2406.63 | 2423.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:45:00 | 2533.70 | 2406.63 | 2423.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2514.60 | 2438.15 | 2437.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.44 | 2440.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.69 | 2683.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 2794.50 | 2797.69 | 2683.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.00 | 2805.32 | 2754.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:45:00 | 2758.60 | 2805.32 | 2754.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 2756.70 | 2804.84 | 2754.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 2753.50 | 2804.84 | 2754.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 2765.90 | 2804.45 | 2754.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:45:00 | 2778.50 | 2804.20 | 2754.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 15:15:00 | 2736.80 | 2802.03 | 2754.57 | SL hit (close<static) qty=1.00 sl=2750.30 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.80 | 2720.57 | 2721.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2374.00 | 2700.46 | 2710.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2263.38 | 2367.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 10:15:00 | 2264.60 | 2263.38 | 2367.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 11:00:00 | 2269.10 | 2263.44 | 2366.74 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 14:00:00 | 2276.00 | 2263.52 | 2365.24 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-08 14:30:00 | 2272.50 | 2263.73 | 2364.84 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | SL hit (close>ema400) qty=1.00 sl=2361.24 alert=retest1 |

### Cycle 6 — BUY (started 2026-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 12:15:00 | 2552.30 | 2410.96 | 2410.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 2557.00 | 2415.95 | 2413.37 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-27 14:00:00 | 2329.60 | 2025-06-11 09:15:00 | 2213.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-27 14:00:00 | 2329.60 | 2025-06-18 14:15:00 | 2280.60 | STOP_HIT | 0.50 | 2.10% |
| SELL | retest2 | 2025-06-30 10:30:00 | 2334.70 | 2025-07-01 09:15:00 | 2369.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-30 12:00:00 | 2334.90 | 2025-07-01 09:15:00 | 2369.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-07-28 14:30:00 | 2353.00 | 2025-08-18 10:15:00 | 2588.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 13:00:00 | 2350.50 | 2025-08-18 10:15:00 | 2585.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-29 14:00:00 | 2398.10 | 2025-10-01 09:15:00 | 2322.70 | STOP_HIT | 1.00 | -3.14% |
| BUY | retest2 | 2025-09-29 09:30:00 | 2350.70 | 2025-10-01 09:15:00 | 2322.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-01-19 10:45:00 | 2778.50 | 2026-01-19 15:15:00 | 2736.80 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-23 11:00:00 | 2782.00 | 2026-01-23 13:15:00 | 2729.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest1 | 2026-04-08 10:15:00 | 2264.60 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest1 | 2026-04-08 11:00:00 | 2269.10 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest1 | 2026-04-08 14:00:00 | 2276.00 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest1 | 2026-04-08 14:30:00 | 2272.50 | 2026-04-10 09:15:00 | 2364.60 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2026-04-13 09:15:00 | 2300.60 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -4.97% |
| SELL | retest2 | 2026-04-13 11:30:00 | 2344.70 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2026-04-13 15:15:00 | 2348.00 | 2026-04-15 09:15:00 | 2414.90 | STOP_HIT | 1.00 | -2.85% |
