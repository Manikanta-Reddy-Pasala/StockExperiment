# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3032.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 16 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 23 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 3
- **Avg / median % per leg:** 1.26% / -1.11%
- **Sum % (uncompounded):** 28.93%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 1 | 5.9% | 1 | 16 | 0 | -0.95% | -16.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 17 | 1 | 5.9% | 1 | 16 | 0 | -0.95% | -16.1% |
| SELL (all) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 6 | 100.0% | 3 | 0 | 3 | 7.50% | 45.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 23 | 7 | 30.4% | 4 | 16 | 3 | 1.26% | 28.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 3001.00 | 2876.90 | 2876.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3033.20 | 2879.68 | 2877.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 2978.00 | 2992.28 | 2948.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 2978.00 | 2992.28 | 2948.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2940.00 | 2991.29 | 2949.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 3029.00 | 2950.74 | 2938.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 2892.30 | 2974.20 | 2955.19 | SL hit (close<static) qty=1.00 sl=2895.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 2852.10 | 2941.23 | 2941.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 2836.60 | 2940.19 | 2941.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 2917.50 | 2913.41 | 2926.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:30:00 | 2907.50 | 2913.41 | 2926.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2878.70 | 2911.70 | 2924.73 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 3056.30 | 2935.40 | 2934.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 3076.00 | 2944.18 | 2939.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2991.90 | 3004.27 | 2974.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:45:00 | 2986.80 | 3004.27 | 2974.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 12:15:00 | 2958.80 | 3003.68 | 2974.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:00:00 | 2958.80 | 3003.68 | 2974.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 2938.70 | 3003.04 | 2974.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 14:00:00 | 2938.70 | 3003.04 | 2974.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 2989.70 | 3000.59 | 2973.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 3000.00 | 2999.18 | 2973.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:00:00 | 3006.30 | 3011.12 | 2982.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 15:15:00 | 2999.00 | 3013.02 | 2984.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 12:15:00 | 2950.10 | 3011.39 | 2984.37 | SL hit (close<static) qty=1.00 sl=2965.20 alert=retest2 |

### Cycle 4 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 2970.60 | 3099.39 | 3100.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2966.00 | 3096.80 | 3098.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3083.30 | 3082.88 | 3090.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 3083.30 | 3082.88 | 3090.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3083.30 | 3082.88 | 3090.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:30:00 | 3040.00 | 3083.13 | 3090.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 3047.20 | 3082.30 | 3090.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 3033.30 | 3081.56 | 3089.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 2888.00 | 3064.38 | 3080.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 2894.84 | 3064.38 | 3080.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 2881.64 | 3062.48 | 3079.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-03 14:15:00 | 2736.00 | 2962.66 | 3019.14 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 2738.80 | 2571.21 | 2571.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 2742.20 | 2577.78 | 2574.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-27 09:30:00 | 3029.00 | 2025-07-08 10:15:00 | 2892.30 | STOP_HIT | 1.00 | -4.51% |
| BUY | retest2 | 2025-09-01 09:15:00 | 3000.00 | 2025-09-05 12:15:00 | 2950.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-09-03 14:00:00 | 3006.30 | 2025-09-05 12:15:00 | 2950.10 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-09-04 15:15:00 | 2999.00 | 2025-09-05 12:15:00 | 2950.10 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-09-08 12:00:00 | 3017.90 | 2025-09-15 09:15:00 | 3319.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-03 13:45:00 | 3130.00 | 2025-10-14 09:15:00 | 3095.20 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-10-03 14:30:00 | 3134.10 | 2025-10-14 09:15:00 | 3095.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-06 09:30:00 | 3128.90 | 2025-10-30 12:15:00 | 3105.70 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-09 09:30:00 | 3127.90 | 2025-10-30 12:15:00 | 3105.70 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-13 14:45:00 | 3116.00 | 2025-10-30 12:15:00 | 3105.70 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-13 15:15:00 | 3120.00 | 2025-10-30 12:15:00 | 3105.70 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-10-15 10:00:00 | 3119.00 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-10-17 13:45:00 | 3123.10 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-21 13:45:00 | 3141.00 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-10-23 09:15:00 | 3151.40 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-10-27 13:00:00 | 3136.00 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-10-27 15:00:00 | 3137.50 | 2025-10-30 13:15:00 | 3073.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-11-18 09:30:00 | 3040.00 | 2025-11-21 14:15:00 | 2888.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 12:00:00 | 3047.20 | 2025-11-21 14:15:00 | 2894.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 13:45:00 | 3033.30 | 2025-11-21 15:15:00 | 2881.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-18 09:30:00 | 3040.00 | 2025-12-03 14:15:00 | 2736.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 12:00:00 | 3047.20 | 2025-12-03 14:15:00 | 2742.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-18 13:45:00 | 3033.30 | 2025-12-04 09:15:00 | 2729.97 | TARGET_HIT | 0.50 | 10.00% |
