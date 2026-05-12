# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3838.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 4 |
| ALERT3 | 49 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 38
- **Target hits / Stop hits / Partials:** 0 / 39 / 2
- **Avg / median % per leg:** -1.38% / -1.73%
- **Sum % (uncompounded):** -56.42%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.87% | -44.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 0 | 0.0% | 0 | 24 | 0 | -1.87% | -44.8% |
| SELL (all) | 17 | 3 | 17.6% | 0 | 15 | 2 | -0.68% | -11.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 3 | 17.6% | 0 | 15 | 2 | -0.68% | -11.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 41 | 3 | 7.3% | 0 | 39 | 2 | -1.38% | -56.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 3100.50 | 3117.64 | 3117.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 3085.40 | 3117.32 | 3117.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 3117.10 | 3114.59 | 3116.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3120.90 | 3114.65 | 3116.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3111.00 | 3114.65 | 3116.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3125.00 | 3114.76 | 3116.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 3101.60 | 3114.66 | 3116.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 2946.52 | 3111.59 | 3114.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 3124.00 | 3111.72 | 3114.55 | SL hit (close>ema200) qty=0.50 sl=3111.72 alert=retest2 |

### Cycle 2 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 3176.70 | 3117.14 | 3117.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 3220.60 | 3123.00 | 3120.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3091.70 | 3135.55 | 3127.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 3091.70 | 3135.55 | 3127.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3091.70 | 3135.55 | 3127.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 3062.10 | 3135.55 | 3127.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3076.00 | 3134.96 | 3126.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 3076.90 | 3134.96 | 3126.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3132.00 | 3121.92 | 3120.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 14:45:00 | 3137.00 | 3122.18 | 3120.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 3153.00 | 3122.82 | 3121.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3106.20 | 3135.33 | 3128.09 | SL hit (close<static) qty=1.00 sl=3116.10 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 3117.10 | 3156.86 | 3157.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 3104.80 | 3156.34 | 3156.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 3165.00 | 3154.33 | 3155.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 15:15:00 | 3165.00 | 3154.33 | 3155.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3165.00 | 3154.33 | 3155.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:45:00 | 3137.00 | 3154.20 | 3155.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 3136.50 | 3154.27 | 3155.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 12:15:00 | 3135.30 | 3154.18 | 3155.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 3138.10 | 3152.95 | 3154.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3179.60 | 3112.88 | 3131.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-17 09:15:00 | 3179.60 | 3112.88 | 3131.21 | SL hit (close>static) qty=1.00 sl=3165.40 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 3305.00 | 3147.23 | 3146.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 3323.70 | 3152.12 | 3149.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 3196.30 | 3221.88 | 3189.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 14:00:00 | 3196.30 | 3221.88 | 3189.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3153.50 | 3220.74 | 3189.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 3129.00 | 3220.74 | 3189.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3142.40 | 3219.96 | 3189.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 3142.80 | 3219.96 | 3189.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 3170.00 | 3212.16 | 3187.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 3170.00 | 3212.16 | 3187.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 3175.40 | 3211.79 | 3187.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 3167.20 | 3211.79 | 3187.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3176.90 | 3210.18 | 3187.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 3176.90 | 3210.18 | 3187.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 3158.90 | 3209.67 | 3186.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 3158.00 | 3209.67 | 3186.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 3145.00 | 3204.54 | 3185.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 3145.00 | 3204.54 | 3185.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 3110.90 | 3170.38 | 3170.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 3099.50 | 3167.86 | 3169.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3134.80 | 3131.15 | 3147.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 3134.80 | 3131.15 | 3147.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3130.90 | 3131.18 | 3147.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 3118.30 | 3130.85 | 3147.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 2962.39 | 3119.85 | 3140.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 3055.40 | 3014.78 | 3071.42 | SL hit (close>ema200) qty=0.50 sl=3014.78 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 3181.00 | 3104.79 | 3104.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 3226.20 | 3115.95 | 3110.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 12:15:00 | 3192.00 | 3195.95 | 3156.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-04 12:30:00 | 3182.20 | 3195.95 | 3156.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 3158.50 | 3195.52 | 3156.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 15:00:00 | 3158.50 | 3195.52 | 3156.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 3150.00 | 3195.07 | 3156.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 3183.30 | 3195.07 | 3156.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 10:00:00 | 3195.60 | 3201.97 | 3163.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 3116.50 | 3220.17 | 3179.37 | SL hit (close<static) qty=1.00 sl=3141.70 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 2963.70 | 3151.96 | 3152.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2937.50 | 3149.83 | 3151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 3360.20 | 3149.48 | 3148.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 3504.10 | 3161.01 | 3154.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-24 09:15:00 | 3142.20 | 2025-07-14 11:15:00 | 3077.40 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-06-25 14:30:00 | 3144.90 | 2025-07-14 11:15:00 | 3077.40 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-06-26 10:30:00 | 3151.00 | 2025-07-14 11:15:00 | 3077.40 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-06-26 13:00:00 | 3153.40 | 2025-07-14 11:15:00 | 3077.40 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-15 11:30:00 | 3151.40 | 2025-07-16 09:15:00 | 3096.10 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-15 15:00:00 | 3147.60 | 2025-07-16 09:15:00 | 3096.10 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-21 10:45:00 | 3149.00 | 2025-07-24 09:15:00 | 3107.40 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-07-21 12:00:00 | 3147.60 | 2025-07-24 09:15:00 | 3107.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-08-08 11:15:00 | 3101.60 | 2025-08-11 09:15:00 | 2946.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-08 11:15:00 | 3101.60 | 2025-08-11 10:15:00 | 3124.00 | STOP_HIT | 0.50 | -0.72% |
| SELL | retest2 | 2025-08-11 10:45:00 | 3101.70 | 2025-08-13 09:15:00 | 3155.50 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-08-11 11:30:00 | 3097.40 | 2025-08-13 09:15:00 | 3155.50 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-08-11 14:15:00 | 3100.80 | 2025-08-13 09:15:00 | 3155.50 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-08-12 10:30:00 | 3105.20 | 2025-08-14 13:15:00 | 3176.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-08-13 11:45:00 | 3110.40 | 2025-08-14 13:15:00 | 3176.70 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-08-13 12:30:00 | 3111.30 | 2025-08-14 13:15:00 | 3176.70 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-09-01 14:45:00 | 3137.00 | 2025-09-05 11:15:00 | 3106.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-09-02 09:30:00 | 3153.00 | 2025-09-05 11:15:00 | 3106.20 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-09-08 12:45:00 | 3138.30 | 2025-09-08 13:15:00 | 3114.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-09 12:30:00 | 3155.50 | 2025-09-26 11:15:00 | 3115.10 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-10-24 10:30:00 | 3166.60 | 2025-10-24 12:15:00 | 3145.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-24 15:15:00 | 3160.00 | 2025-10-28 11:15:00 | 3117.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-10-27 12:00:00 | 3166.60 | 2025-10-28 11:15:00 | 3117.10 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-30 09:45:00 | 3137.00 | 2025-11-17 09:15:00 | 3179.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-30 10:30:00 | 3136.50 | 2025-11-17 09:15:00 | 3179.60 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-10-30 12:15:00 | 3135.30 | 2025-11-17 09:15:00 | 3179.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-31 10:15:00 | 3138.10 | 2025-11-17 09:15:00 | 3179.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-08 10:30:00 | 3118.30 | 2026-01-12 10:15:00 | 2962.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:30:00 | 3118.30 | 2026-01-29 11:15:00 | 3055.40 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2026-02-09 09:15:00 | 3073.00 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-09 10:30:00 | 3110.20 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-02-09 11:00:00 | 3103.30 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-03-05 09:15:00 | 3183.30 | 2026-03-16 10:15:00 | 3116.50 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-09 10:00:00 | 3195.60 | 2026-03-16 10:15:00 | 3116.50 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2026-03-17 11:45:00 | 3173.00 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-03-17 14:45:00 | 3180.30 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2026-03-18 10:30:00 | 3215.00 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-03-18 12:15:00 | 3206.90 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-03-18 13:00:00 | 3209.50 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-03-18 14:45:00 | 3213.40 | 2026-03-19 11:15:00 | 3117.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-03-20 11:15:00 | 3180.50 | 2026-03-20 15:15:00 | 3121.10 | STOP_HIT | 1.00 | -1.87% |
