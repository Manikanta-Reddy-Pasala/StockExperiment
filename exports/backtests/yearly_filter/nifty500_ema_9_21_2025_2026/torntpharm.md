# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4385.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 79 |
| ALERT1 | 59 |
| ALERT2 | 59 |
| ALERT2_SKIP | 28 |
| ALERT3 | 162 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 77 |
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 18 / 60
- **Target hits / Stop hits / Partials:** 1 / 75 / 2
- **Avg / median % per leg:** -0.11% / -0.75%
- **Sum % (uncompounded):** -8.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 8 | 17.8% | 1 | 44 | 0 | -0.15% | -6.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| BUY @ 3rd Alert (retest2) | 44 | 8 | 18.2% | 1 | 43 | 0 | -0.14% | -6.0% |
| SELL (all) | 33 | 10 | 30.3% | 0 | 31 | 2 | -0.06% | -1.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 10 | 30.3% | 0 | 31 | 2 | -0.06% | -1.9% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.94% | -0.9% |
| retest2 (combined) | 77 | 18 | 23.4% | 1 | 74 | 2 | -0.10% | -7.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 15:15:00 | 3228.70 | 3202.15 | 3201.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 3253.00 | 3212.32 | 3206.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 15:15:00 | 3225.00 | 3225.65 | 3217.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 09:15:00 | 3245.00 | 3225.65 | 3217.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 3231.60 | 3226.84 | 3218.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 3261.60 | 3234.25 | 3228.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 15:00:00 | 3264.10 | 3240.22 | 3231.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:00:00 | 3265.00 | 3249.94 | 3237.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 3267.80 | 3257.33 | 3246.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 3296.20 | 3265.60 | 3252.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:30:00 | 3304.00 | 3271.68 | 3256.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 3309.50 | 3276.48 | 3259.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 3304.40 | 3282.07 | 3263.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:45:00 | 3308.90 | 3286.45 | 3267.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3243.30 | 3282.25 | 3270.70 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-20 09:15:00 | 3243.30 | 3282.25 | 3270.70 | SL hit (close<static) qty=1.00 sl=3250.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 3244.60 | 3265.76 | 3266.12 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3354.00 | 3278.18 | 3271.38 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 3233.00 | 3267.39 | 3271.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 14:15:00 | 3211.30 | 3242.07 | 3256.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 13:15:00 | 3179.00 | 3172.74 | 3196.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 14:00:00 | 3179.00 | 3172.74 | 3196.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 3172.10 | 3168.43 | 3180.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 3155.90 | 3168.43 | 3180.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 3165.40 | 3160.23 | 3170.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 15:00:00 | 3163.00 | 3160.79 | 3170.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 13:15:00 | 3162.40 | 3160.47 | 3165.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 3173.30 | 3163.03 | 3166.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 3173.30 | 3163.03 | 3166.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 3158.50 | 3162.13 | 3165.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:15:00 | 3150.10 | 3162.13 | 3165.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 3150.10 | 3159.72 | 3164.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 3129.20 | 3162.87 | 3163.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 3140.20 | 3133.44 | 3143.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 3147.40 | 3131.79 | 3133.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 3147.40 | 3134.91 | 3134.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 3147.40 | 3134.91 | 3134.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 3155.70 | 3145.65 | 3141.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 3203.80 | 3208.98 | 3188.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 09:45:00 | 3196.20 | 3208.98 | 3188.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3194.90 | 3205.42 | 3190.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:45:00 | 3193.30 | 3205.42 | 3190.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 3203.40 | 3202.69 | 3191.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:30:00 | 3210.00 | 3204.13 | 3193.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 3209.90 | 3204.13 | 3193.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 3224.30 | 3244.03 | 3246.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 3224.30 | 3244.03 | 3246.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 12:15:00 | 3216.90 | 3238.60 | 3243.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 3186.20 | 3185.72 | 3206.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 3186.20 | 3185.72 | 3206.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3183.20 | 3168.56 | 3176.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 3183.20 | 3168.56 | 3176.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3175.00 | 3169.85 | 3176.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 3168.30 | 3169.85 | 3176.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 3182.60 | 3172.40 | 3177.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:30:00 | 3179.00 | 3172.40 | 3177.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 12:15:00 | 3182.70 | 3174.46 | 3177.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:00:00 | 3182.70 | 3174.46 | 3177.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 3182.40 | 3176.05 | 3177.98 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 3196.10 | 3180.20 | 3179.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 3214.70 | 3187.10 | 3182.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 3181.10 | 3205.65 | 3196.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 3181.10 | 3205.65 | 3196.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3189.20 | 3202.36 | 3195.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 3185.60 | 3202.36 | 3195.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 3188.20 | 3199.53 | 3195.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 3188.20 | 3199.53 | 3195.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 3196.50 | 3198.92 | 3195.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 3178.60 | 3198.92 | 3195.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 3208.00 | 3200.74 | 3196.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:15:00 | 3212.50 | 3200.74 | 3196.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 3322.00 | 3358.34 | 3360.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 3322.00 | 3358.34 | 3360.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 10:15:00 | 3314.80 | 3349.63 | 3355.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 13:15:00 | 3342.60 | 3342.49 | 3350.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:00:00 | 3342.60 | 3342.49 | 3350.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 3356.00 | 3342.95 | 3348.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:15:00 | 3374.80 | 3342.95 | 3348.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 3374.20 | 3349.20 | 3350.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:30:00 | 3384.20 | 3349.20 | 3350.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 3368.30 | 3353.02 | 3352.46 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 3317.00 | 3349.23 | 3353.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 3302.30 | 3339.85 | 3348.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 3345.40 | 3330.65 | 3338.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:00:00 | 3345.40 | 3330.65 | 3338.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 3327.80 | 3330.08 | 3337.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 3341.70 | 3330.08 | 3337.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 13:15:00 | 3340.50 | 3333.05 | 3337.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:45:00 | 3347.20 | 3333.05 | 3337.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 14:15:00 | 3357.60 | 3337.96 | 3339.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 15:00:00 | 3357.60 | 3337.96 | 3339.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 3317.50 | 3334.48 | 3337.39 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 3344.00 | 3338.89 | 3338.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 09:15:00 | 3348.70 | 3340.86 | 3339.55 | Break + close above crossover candle high |

### Cycle 12 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 3329.40 | 3338.56 | 3338.62 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 3339.10 | 3338.67 | 3338.67 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 3322.00 | 3335.34 | 3337.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 3313.00 | 3329.10 | 3333.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3351.60 | 3330.64 | 3333.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 3351.10 | 3330.64 | 3333.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 3367.30 | 3337.97 | 3336.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 3387.10 | 3347.80 | 3341.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 3428.70 | 3434.04 | 3410.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:00:00 | 3428.70 | 3434.04 | 3410.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3476.00 | 3512.30 | 3505.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 3476.00 | 3512.30 | 3505.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3475.10 | 3504.86 | 3502.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 3475.10 | 3504.86 | 3502.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 3491.90 | 3500.62 | 3500.83 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 3504.60 | 3501.42 | 3501.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 15:15:00 | 3510.90 | 3503.79 | 3502.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 3542.80 | 3542.97 | 3528.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 3546.80 | 3542.97 | 3528.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 3542.00 | 3542.77 | 3529.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 3573.40 | 3532.29 | 3528.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 3649.10 | 3700.13 | 3701.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 3649.10 | 3700.13 | 3701.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 3630.40 | 3671.56 | 3686.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 10:15:00 | 3552.00 | 3548.77 | 3578.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 10:30:00 | 3551.40 | 3548.77 | 3578.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3595.80 | 3564.90 | 3576.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 3595.80 | 3564.90 | 3576.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3595.00 | 3570.92 | 3578.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 3582.40 | 3570.92 | 3578.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 12:15:00 | 3599.60 | 3584.93 | 3583.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 12:15:00 | 3599.60 | 3584.93 | 3583.61 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 3567.50 | 3583.09 | 3583.11 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 11:15:00 | 3596.10 | 3584.43 | 3583.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 12:15:00 | 3600.00 | 3587.55 | 3584.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 3600.30 | 3600.79 | 3593.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 3600.30 | 3600.79 | 3593.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 3597.00 | 3600.03 | 3593.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:30:00 | 3594.60 | 3600.03 | 3593.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 3625.00 | 3605.03 | 3596.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 11:45:00 | 3642.00 | 3624.38 | 3616.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 14:15:00 | 3645.10 | 3623.24 | 3619.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:00:00 | 3642.00 | 3636.44 | 3627.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 3646.40 | 3637.18 | 3630.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 3665.50 | 3659.41 | 3648.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 3643.60 | 3659.41 | 3648.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 3660.10 | 3659.55 | 3649.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 3632.20 | 3659.55 | 3649.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 3668.50 | 3665.95 | 3656.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 3660.20 | 3665.95 | 3656.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 3648.60 | 3662.48 | 3655.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 3648.60 | 3662.48 | 3655.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 3651.70 | 3660.33 | 3655.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 3664.00 | 3660.33 | 3655.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 3650.50 | 3659.41 | 3655.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 3650.50 | 3659.41 | 3655.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 3656.80 | 3658.89 | 3655.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:45:00 | 3668.40 | 3660.63 | 3657.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 3678.00 | 3671.66 | 3665.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 3633.00 | 3664.72 | 3664.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 3633.00 | 3664.72 | 3664.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 3596.00 | 3628.88 | 3645.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 3606.40 | 3592.02 | 3610.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 3606.40 | 3592.02 | 3610.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3586.90 | 3577.99 | 3592.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 3586.90 | 3577.99 | 3592.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 3599.60 | 3581.79 | 3591.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 3599.60 | 3581.79 | 3591.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 3597.50 | 3584.93 | 3592.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:45:00 | 3594.40 | 3584.93 | 3592.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 3584.10 | 3584.76 | 3591.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:15:00 | 3599.90 | 3584.76 | 3591.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 3599.90 | 3587.79 | 3592.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 3547.50 | 3587.79 | 3592.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 3568.00 | 3583.83 | 3590.18 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 3606.50 | 3589.39 | 3589.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 13:15:00 | 3609.90 | 3600.93 | 3596.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 3594.10 | 3599.57 | 3596.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 3594.10 | 3599.57 | 3596.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 3596.20 | 3598.89 | 3596.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:30:00 | 3614.80 | 3603.53 | 3598.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3577.60 | 3598.12 | 3597.26 | SL hit (close<static) qty=1.00 sl=3591.30 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 12:15:00 | 3573.10 | 3593.11 | 3595.06 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 3606.20 | 3595.64 | 3594.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 13:15:00 | 3627.50 | 3604.71 | 3599.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3593.10 | 3603.51 | 3600.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:30:00 | 3595.50 | 3603.51 | 3600.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 3615.00 | 3605.80 | 3601.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 3607.50 | 3605.80 | 3601.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 3592.00 | 3621.09 | 3614.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 3592.00 | 3621.09 | 3614.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3574.50 | 3611.77 | 3610.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 3574.50 | 3611.77 | 3610.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 3581.90 | 3605.80 | 3608.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 3560.00 | 3577.47 | 3586.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 11:15:00 | 3540.90 | 3539.79 | 3554.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 11:30:00 | 3540.10 | 3539.79 | 3554.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 3556.30 | 3545.15 | 3554.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:30:00 | 3551.50 | 3545.15 | 3554.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 3536.00 | 3543.32 | 3552.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3521.70 | 3536.76 | 3548.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 3522.80 | 3533.25 | 3545.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 3519.20 | 3533.25 | 3545.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 12:15:00 | 3556.80 | 3546.27 | 3545.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 3556.80 | 3546.27 | 3545.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 3565.30 | 3550.07 | 3547.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 14:15:00 | 3650.70 | 3661.04 | 3638.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 15:00:00 | 3650.70 | 3661.04 | 3638.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3624.50 | 3651.34 | 3638.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 3624.50 | 3651.34 | 3638.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3612.00 | 3643.47 | 3635.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:30:00 | 3610.30 | 3643.47 | 3635.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 3612.60 | 3630.14 | 3631.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 3593.60 | 3621.16 | 3626.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 12:15:00 | 3558.30 | 3552.49 | 3578.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 12:30:00 | 3553.90 | 3552.49 | 3578.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 3564.40 | 3554.87 | 3576.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 13:45:00 | 3578.80 | 3554.87 | 3576.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 3587.90 | 3563.10 | 3576.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3594.00 | 3571.02 | 3579.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3593.20 | 3575.46 | 3580.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 3580.90 | 3576.92 | 3580.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 3589.90 | 3583.70 | 3583.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 14:15:00 | 3589.90 | 3583.70 | 3583.18 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 3558.00 | 3580.37 | 3581.87 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 13:15:00 | 3608.40 | 3581.49 | 3581.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 09:15:00 | 3624.10 | 3594.54 | 3587.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3585.00 | 3592.80 | 3588.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 3585.00 | 3592.80 | 3588.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 3575.10 | 3589.26 | 3586.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 3575.10 | 3589.26 | 3586.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 3564.30 | 3582.69 | 3584.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 09:15:00 | 3544.00 | 3572.19 | 3579.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-06 14:15:00 | 3521.90 | 3513.73 | 3531.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-06 15:00:00 | 3521.90 | 3513.73 | 3531.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3492.00 | 3511.01 | 3527.58 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 12:15:00 | 3548.60 | 3529.11 | 3526.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 13:15:00 | 3560.60 | 3535.41 | 3529.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 3538.60 | 3545.08 | 3538.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 3538.60 | 3545.08 | 3538.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 3548.70 | 3545.81 | 3539.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 3545.70 | 3545.81 | 3539.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3532.00 | 3543.41 | 3540.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 3532.00 | 3543.41 | 3540.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3537.50 | 3542.23 | 3539.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:15:00 | 3547.50 | 3542.23 | 3539.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 15:15:00 | 3545.70 | 3546.72 | 3543.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:30:00 | 3542.40 | 3544.03 | 3543.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 3536.80 | 3541.49 | 3542.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3536.80 | 3541.49 | 3542.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 3490.80 | 3524.63 | 3533.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3526.50 | 3516.50 | 3525.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 3526.50 | 3516.50 | 3525.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3518.50 | 3516.90 | 3524.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3520.70 | 3516.90 | 3524.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3540.60 | 3521.64 | 3525.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 3540.60 | 3521.64 | 3525.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 3545.10 | 3526.33 | 3527.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 3545.10 | 3526.33 | 3527.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 3554.00 | 3531.87 | 3530.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 3570.00 | 3540.89 | 3536.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 3568.60 | 3578.18 | 3566.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 3568.60 | 3578.18 | 3566.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 3569.50 | 3576.45 | 3566.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3565.20 | 3574.96 | 3566.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 3568.00 | 3573.57 | 3567.00 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2025-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 12:15:00 | 3564.80 | 3586.24 | 3588.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 3557.90 | 3580.57 | 3585.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 3586.80 | 3579.75 | 3583.98 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 3598.00 | 3586.33 | 3585.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 3605.80 | 3590.22 | 3587.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3578.00 | 3587.78 | 3586.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 3560.30 | 3587.78 | 3586.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3596.90 | 3589.60 | 3587.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:30:00 | 3598.20 | 3591.32 | 3588.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 3598.20 | 3591.32 | 3588.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 14:30:00 | 3599.90 | 3594.38 | 3590.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 3604.00 | 3594.38 | 3590.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 3585.20 | 3594.86 | 3591.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 3585.20 | 3594.86 | 3591.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 3590.00 | 3593.89 | 3591.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 3582.80 | 3593.89 | 3591.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 3590.00 | 3593.11 | 3591.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 3562.60 | 3586.37 | 3588.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 13:15:00 | 3562.60 | 3586.37 | 3588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 3545.50 | 3578.19 | 3584.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 3595.30 | 3576.66 | 3582.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 3595.30 | 3576.66 | 3582.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3591.80 | 3579.69 | 3583.43 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-11-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 12:15:00 | 3618.20 | 3589.82 | 3587.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 3626.50 | 3597.16 | 3591.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 3605.00 | 3611.31 | 3601.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:45:00 | 3604.40 | 3611.31 | 3601.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 3603.40 | 3609.73 | 3602.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 3600.10 | 3609.73 | 3602.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 13:15:00 | 3581.60 | 3604.10 | 3600.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:00:00 | 3581.60 | 3604.10 | 3600.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 3574.60 | 3598.20 | 3597.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 15:00:00 | 3574.60 | 3598.20 | 3597.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 3573.00 | 3593.16 | 3595.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 3550.50 | 3584.63 | 3591.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 3565.00 | 3563.00 | 3573.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 3565.00 | 3563.00 | 3573.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3580.90 | 3566.58 | 3574.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3585.40 | 3566.58 | 3574.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3581.10 | 3569.49 | 3574.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 3581.10 | 3569.49 | 3574.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3581.00 | 3573.82 | 3575.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3727.00 | 3573.82 | 3575.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3790.80 | 3617.22 | 3595.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 3813.00 | 3683.04 | 3630.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 3817.80 | 3821.07 | 3768.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 3822.40 | 3837.95 | 3822.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 3822.40 | 3837.95 | 3822.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:30:00 | 3830.20 | 3837.95 | 3822.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3818.30 | 3834.02 | 3822.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 3817.10 | 3834.02 | 3822.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3827.10 | 3836.35 | 3826.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 3827.10 | 3836.35 | 3826.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3817.20 | 3832.52 | 3825.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:30:00 | 3813.40 | 3832.52 | 3825.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 3819.30 | 3829.87 | 3825.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 3819.00 | 3829.87 | 3825.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 3814.10 | 3822.30 | 3822.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 3791.30 | 3816.10 | 3819.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 3715.00 | 3708.69 | 3728.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 3715.00 | 3708.69 | 3728.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 3725.20 | 3711.99 | 3727.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 3707.00 | 3720.38 | 3727.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 3742.00 | 3728.02 | 3729.21 | SL hit (close>static) qty=1.00 sl=3740.90 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 3748.30 | 3733.58 | 3731.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 3754.60 | 3740.72 | 3735.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 3738.80 | 3748.60 | 3743.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 3737.60 | 3748.60 | 3743.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 3729.60 | 3744.80 | 3742.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 3729.60 | 3744.80 | 3742.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3728.50 | 3741.54 | 3740.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:15:00 | 3725.40 | 3741.54 | 3740.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 3725.40 | 3738.31 | 3739.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 3724.10 | 3732.64 | 3736.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 3721.00 | 3720.69 | 3726.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:00:00 | 3721.00 | 3720.69 | 3726.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 3732.10 | 3722.97 | 3726.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 3732.10 | 3722.97 | 3726.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 3733.30 | 3725.04 | 3727.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 3734.60 | 3725.04 | 3727.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 3739.30 | 3727.89 | 3728.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:30:00 | 3741.00 | 3727.89 | 3728.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 3732.00 | 3728.71 | 3728.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:15:00 | 3726.60 | 3728.71 | 3728.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 3711.40 | 3725.25 | 3727.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 10:30:00 | 3704.30 | 3721.14 | 3725.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 3745.20 | 3725.30 | 3724.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 09:15:00 | 3745.20 | 3725.30 | 3724.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 3771.30 | 3742.36 | 3734.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 3759.50 | 3772.98 | 3755.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 3759.50 | 3772.98 | 3755.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 3788.90 | 3776.17 | 3758.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 11:30:00 | 3792.20 | 3779.39 | 3761.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:45:00 | 3793.00 | 3782.44 | 3766.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 15:00:00 | 3789.80 | 3783.91 | 3768.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 3757.30 | 3768.78 | 3766.64 | SL hit (close<static) qty=1.00 sl=3758.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 3753.00 | 3764.08 | 3764.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 3744.90 | 3760.24 | 3762.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 3758.30 | 3756.76 | 3760.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 3758.30 | 3756.76 | 3760.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 3774.10 | 3760.23 | 3761.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 3774.10 | 3760.23 | 3761.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 3763.90 | 3760.96 | 3762.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 3771.00 | 3760.96 | 3762.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3769.30 | 3762.63 | 3762.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3769.30 | 3762.63 | 3762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 3764.00 | 3762.90 | 3762.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 11:15:00 | 3788.10 | 3768.12 | 3765.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 3767.00 | 3779.84 | 3773.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 15:00:00 | 3800.00 | 3785.20 | 3778.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 13:15:00 | 3776.60 | 3781.73 | 3782.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 3776.60 | 3781.73 | 3782.10 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 3789.90 | 3783.08 | 3782.64 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 3775.40 | 3781.55 | 3781.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 3762.30 | 3777.19 | 3779.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 3770.70 | 3770.65 | 3775.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 15:15:00 | 3745.10 | 3770.65 | 3775.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 09:45:00 | 3747.00 | 3760.55 | 3770.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 3740.50 | 3756.49 | 3762.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:30:00 | 3747.60 | 3752.15 | 3754.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 3752.00 | 3752.12 | 3754.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 3752.00 | 3752.12 | 3754.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 3761.30 | 3753.95 | 3754.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 3761.30 | 3753.95 | 3754.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 3798.90 | 3762.94 | 3758.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 3798.90 | 3762.94 | 3758.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 10:15:00 | 3810.10 | 3782.40 | 3769.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 3811.10 | 3815.17 | 3801.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 3805.70 | 3813.28 | 3802.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 3805.70 | 3813.28 | 3802.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 3809.50 | 3813.28 | 3802.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 3806.10 | 3822.81 | 3813.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:00:00 | 3806.10 | 3822.81 | 3813.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 3812.40 | 3820.73 | 3813.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:45:00 | 3821.50 | 3816.90 | 3813.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 15:15:00 | 3835.00 | 3816.24 | 3813.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 10:15:00 | 3802.20 | 3813.96 | 3813.22 | SL hit (close<static) qty=1.00 sl=3804.30 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 3799.70 | 3811.10 | 3811.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 13:15:00 | 3779.90 | 3803.23 | 3808.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 3786.90 | 3782.58 | 3794.18 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 3859.40 | 3802.85 | 3800.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 3871.70 | 3846.01 | 3832.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 3862.00 | 3871.76 | 3859.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 3862.00 | 3871.76 | 3859.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 3873.00 | 3872.01 | 3860.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:45:00 | 3873.40 | 3872.01 | 3860.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 3872.90 | 3872.19 | 3861.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 3881.30 | 3872.19 | 3861.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:45:00 | 3884.50 | 3878.35 | 3867.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 15:15:00 | 3955.00 | 3985.98 | 3990.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 15:15:00 | 3955.00 | 3985.98 | 3990.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 3923.40 | 3973.46 | 3984.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 3953.10 | 3945.25 | 3963.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 3953.10 | 3945.25 | 3963.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3940.00 | 3944.20 | 3961.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 3926.30 | 3944.20 | 3961.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:15:00 | 3933.20 | 3939.01 | 3955.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 3925.00 | 3931.09 | 3947.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 3933.10 | 3933.58 | 3947.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3932.50 | 3933.28 | 3944.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:15:00 | 3945.20 | 3933.28 | 3944.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 3953.60 | 3937.35 | 3945.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 3953.60 | 3937.35 | 3945.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 3958.30 | 3941.54 | 3946.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:45:00 | 3983.90 | 3941.54 | 3946.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 3966.60 | 3946.55 | 3948.49 | SL hit (close>static) qty=1.00 sl=3961.90 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 3987.40 | 3954.72 | 3952.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 14:15:00 | 4003.30 | 3964.44 | 3956.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4038.30 | 4055.71 | 4028.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 4039.30 | 4055.71 | 4028.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4030.50 | 4050.66 | 4029.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 11:30:00 | 4043.70 | 4048.87 | 4030.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 12:15:00 | 4021.30 | 4043.36 | 4029.36 | SL hit (close<static) qty=1.00 sl=4023.80 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 3975.10 | 4013.28 | 4017.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 10:15:00 | 3948.20 | 3994.46 | 4008.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 3994.60 | 3990.54 | 4003.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 3994.60 | 3990.54 | 4003.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 3950.10 | 3974.94 | 3992.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 4020.50 | 3984.35 | 3995.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 4010.90 | 3989.66 | 3996.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 4010.90 | 3989.66 | 3996.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 4000.00 | 3992.10 | 3996.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:45:00 | 4004.50 | 3992.10 | 3996.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 4010.70 | 3995.82 | 3997.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 13:45:00 | 4014.10 | 3995.82 | 3997.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 4020.90 | 4000.84 | 3999.91 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 3952.70 | 3993.86 | 3997.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 3922.60 | 3979.60 | 3990.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 12:15:00 | 3973.30 | 3965.24 | 3977.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 13:00:00 | 3973.30 | 3965.24 | 3977.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 3966.80 | 3965.56 | 3976.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 3950.10 | 3969.54 | 3976.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 3752.59 | 3932.64 | 3944.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 3943.70 | 3934.85 | 3944.58 | SL hit (close>ema200) qty=0.50 sl=3934.85 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 3988.10 | 3953.10 | 3951.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 4001.90 | 3962.54 | 3956.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 3972.60 | 3979.72 | 3968.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:15:00 | 3959.20 | 3979.72 | 3968.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 3943.30 | 3972.44 | 3966.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 3940.60 | 3972.44 | 3966.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3913.90 | 3960.73 | 3961.44 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 3984.90 | 3960.25 | 3960.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 3993.00 | 3966.80 | 3963.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3998.70 | 4019.99 | 4001.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 4006.80 | 4019.99 | 4001.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 4017.90 | 4019.57 | 4003.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:15:00 | 4025.00 | 4019.57 | 4003.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:15:00 | 4025.20 | 4018.78 | 4004.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-05 09:15:00 | 3969.90 | 4011.38 | 4006.12 | SL hit (close<static) qty=1.00 sl=3986.70 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 11:15:00 | 3982.80 | 4001.64 | 4002.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 12:15:00 | 3977.10 | 3996.73 | 4000.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 3974.00 | 3952.28 | 3966.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 3974.00 | 3952.28 | 3966.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4011.00 | 3964.02 | 3970.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 4011.00 | 3964.02 | 3970.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 13:15:00 | 3995.00 | 3978.72 | 3976.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 4005.00 | 3986.76 | 3980.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 4053.60 | 4056.59 | 4033.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 12:45:00 | 4054.60 | 4056.59 | 4033.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 4038.10 | 4052.05 | 4038.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:45:00 | 4032.20 | 4052.05 | 4038.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 4050.40 | 4051.72 | 4039.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:45:00 | 4063.10 | 4053.47 | 4041.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 12:45:00 | 4065.00 | 4056.18 | 4044.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-26 09:15:00 | 4469.41 | 4421.14 | 4378.90 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 13:15:00 | 4356.60 | 4380.22 | 4382.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 4331.90 | 4370.55 | 4378.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 13:15:00 | 4378.90 | 4356.54 | 4365.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:00:00 | 4378.90 | 4356.54 | 4365.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 4366.70 | 4358.57 | 4365.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:30:00 | 4395.80 | 4358.57 | 4365.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 4365.00 | 4359.86 | 4365.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 4311.30 | 4359.86 | 4365.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 10:15:00 | 4356.10 | 4339.91 | 4346.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 4363.00 | 4351.96 | 4350.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 12:15:00 | 4363.00 | 4351.96 | 4350.86 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 09:15:00 | 4331.00 | 4348.33 | 4349.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 4299.90 | 4338.64 | 4345.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 4338.20 | 4338.13 | 4343.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 4334.70 | 4338.13 | 4343.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 4344.90 | 4320.12 | 4329.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 13:00:00 | 4344.90 | 4320.12 | 4329.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 13:15:00 | 4342.80 | 4324.65 | 4330.35 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 4382.00 | 4336.12 | 4335.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 4424.50 | 4359.22 | 4346.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 4400.50 | 4422.64 | 4406.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 4400.20 | 4422.64 | 4406.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4419.60 | 4422.03 | 4408.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 4440.00 | 4424.13 | 4411.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 4436.80 | 4432.70 | 4419.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 4387.80 | 4420.31 | 4415.66 | SL hit (close<static) qty=1.00 sl=4400.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 4370.40 | 4406.90 | 4410.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 4282.00 | 4373.89 | 4392.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 12:15:00 | 4316.10 | 4295.69 | 4327.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 13:00:00 | 4316.10 | 4295.69 | 4327.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 4341.60 | 4304.87 | 4328.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 4341.60 | 4304.87 | 4328.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 4308.30 | 4305.56 | 4326.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 4302.90 | 4305.56 | 4326.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:30:00 | 4302.30 | 4301.20 | 4320.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 4299.30 | 4246.20 | 4239.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 4299.30 | 4246.20 | 4239.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 4316.30 | 4269.95 | 4252.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4280.40 | 4289.39 | 4270.69 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4225.00 | 4264.68 | 4266.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4201.30 | 4252.01 | 4260.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 4243.40 | 4243.28 | 4253.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 4243.40 | 4243.28 | 4253.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 4221.10 | 4235.18 | 4248.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 4327.20 | 4235.18 | 4248.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4262.80 | 4240.71 | 4249.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 4234.10 | 4240.71 | 4249.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 4022.39 | 4109.98 | 4175.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 3999.80 | 3992.69 | 4053.57 | SL hit (close>ema200) qty=0.50 sl=3992.69 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 4090.20 | 4048.28 | 4042.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 4100.50 | 4058.72 | 4048.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 15:15:00 | 4158.00 | 4171.70 | 4140.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:15:00 | 4199.60 | 4171.70 | 4140.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 4160.00 | 4180.69 | 4161.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-15 14:15:00 | 4160.00 | 4180.69 | 4161.24 | SL hit (close<ema400) qty=1.00 sl=4161.24 alert=retest1 |

### Cycle 72 — SELL (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 11:15:00 | 4117.50 | 4151.28 | 4152.27 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 4166.80 | 4154.11 | 4152.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 13:15:00 | 4172.50 | 4159.09 | 4155.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 11:15:00 | 4165.10 | 4169.55 | 4162.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 12:00:00 | 4165.10 | 4169.55 | 4162.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 4180.10 | 4176.16 | 4169.18 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 14:15:00 | 4135.80 | 4161.94 | 4164.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 15:15:00 | 4121.10 | 4153.77 | 4160.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 4156.40 | 4114.33 | 4130.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 4156.40 | 4114.33 | 4130.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 4156.20 | 4122.70 | 4132.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 4140.70 | 4124.80 | 4132.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 4146.20 | 4137.71 | 4137.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 14:15:00 | 4146.20 | 4137.71 | 4137.43 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 4115.10 | 4133.73 | 4135.70 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 4176.00 | 4133.15 | 4132.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 10:15:00 | 4210.00 | 4148.52 | 4139.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 12:15:00 | 4204.20 | 4214.41 | 4189.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 12:30:00 | 4199.20 | 4214.41 | 4189.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 4183.00 | 4207.84 | 4191.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 4183.00 | 4207.84 | 4191.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 4186.10 | 4203.49 | 4190.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 4246.40 | 4203.49 | 4190.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 4163.80 | 4221.08 | 4213.39 | SL hit (close<static) qty=1.00 sl=4175.10 alert=retest2 |

### Cycle 78 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 4154.70 | 4207.80 | 4208.05 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 4248.30 | 4204.48 | 4203.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 15:15:00 | 4262.90 | 4240.60 | 4225.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 4352.00 | 4360.17 | 4332.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 4372.70 | 4360.17 | 4332.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4386.80 | 4365.50 | 4337.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:15:00 | 4394.50 | 4365.50 | 4337.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 14:30:00 | 4395.00 | 4371.90 | 4351.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 14:00:00 | 3261.60 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-05-15 15:00:00 | 3264.10 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-16 10:00:00 | 3265.00 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-05-16 15:00:00 | 3267.80 | 2025-05-20 09:15:00 | 3243.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-05-19 10:30:00 | 3304.00 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-05-19 12:15:00 | 3309.50 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-05-19 13:00:00 | 3304.40 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-05-19 13:45:00 | 3308.90 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2025-05-20 12:15:00 | 3272.70 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-20 12:45:00 | 3271.40 | 2025-05-20 14:15:00 | 3244.60 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-28 09:15:00 | 3155.90 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-05-28 14:15:00 | 3165.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-05-28 15:00:00 | 3163.00 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-05-29 13:15:00 | 3162.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-06-02 09:15:00 | 3129.20 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-03 09:45:00 | 3140.20 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-06-05 10:00:00 | 3147.40 | 2025-06-05 10:15:00 | 3147.40 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-06-11 14:30:00 | 3210.00 | 2025-06-18 11:15:00 | 3224.30 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-06-11 15:00:00 | 3209.90 | 2025-06-18 11:15:00 | 3224.30 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2025-06-26 14:15:00 | 3212.50 | 2025-07-03 09:15:00 | 3322.00 | STOP_HIT | 1.00 | 3.41% |
| BUY | retest2 | 2025-07-25 09:15:00 | 3573.40 | 2025-08-01 11:15:00 | 3649.10 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2025-08-08 09:15:00 | 3582.40 | 2025-08-08 12:15:00 | 3599.60 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-08-14 11:45:00 | 3642.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-18 14:15:00 | 3645.10 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-08-19 10:00:00 | 3642.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-08-19 13:15:00 | 3646.40 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-08-22 13:45:00 | 3668.40 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-25 11:45:00 | 3678.00 | 2025-08-26 09:15:00 | 3633.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-09-05 09:30:00 | 3614.80 | 2025-09-05 11:15:00 | 3577.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3521.70 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-17 10:30:00 | 3522.80 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-09-17 11:00:00 | 3519.20 | 2025-09-18 12:15:00 | 3556.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-09-29 11:30:00 | 3580.90 | 2025-09-29 14:15:00 | 3589.90 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-10 11:15:00 | 3547.50 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-10-10 15:15:00 | 3545.70 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-13 09:30:00 | 3542.40 | 2025-10-13 11:15:00 | 3536.80 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-10-30 11:30:00 | 3598.20 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-30 12:00:00 | 3598.20 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-30 14:30:00 | 3599.90 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-10-30 15:00:00 | 3604.00 | 2025-10-31 13:15:00 | 3562.60 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-25 09:15:00 | 3707.00 | 2025-11-25 11:15:00 | 3742.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-02 10:30:00 | 3704.30 | 2025-12-03 09:15:00 | 3745.20 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-05 11:30:00 | 3792.20 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-12-05 13:45:00 | 3793.00 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-05 15:00:00 | 3789.80 | 2025-12-08 13:15:00 | 3757.30 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-11 15:00:00 | 3800.00 | 2025-12-15 13:15:00 | 3776.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-12-16 15:15:00 | 3745.10 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-12-17 09:45:00 | 3747.00 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-12-18 09:15:00 | 3740.50 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-12-19 11:30:00 | 3747.60 | 2025-12-19 14:15:00 | 3798.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-12-26 13:45:00 | 3821.50 | 2025-12-29 10:15:00 | 3802.20 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-12-26 15:15:00 | 3835.00 | 2025-12-29 10:15:00 | 3802.20 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 09:15:00 | 3881.30 | 2026-01-09 15:15:00 | 3955.00 | STOP_HIT | 1.00 | 1.90% |
| BUY | retest2 | 2026-01-06 12:45:00 | 3884.50 | 2026-01-09 15:15:00 | 3955.00 | STOP_HIT | 1.00 | 1.81% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3926.30 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-01-13 11:15:00 | 3933.20 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-01-13 13:45:00 | 3925.00 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3933.10 | 2026-01-14 12:15:00 | 3966.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-20 11:30:00 | 4043.70 | 2026-01-20 12:15:00 | 4021.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2026-01-28 10:00:00 | 3950.10 | 2026-01-30 09:15:00 | 3752.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 10:00:00 | 3950.10 | 2026-01-30 10:15:00 | 3943.70 | STOP_HIT | 0.50 | 0.16% |
| SELL | retest2 | 2026-01-30 11:00:00 | 3943.70 | 2026-01-30 13:15:00 | 3988.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-30 11:45:00 | 3956.60 | 2026-01-30 13:15:00 | 3988.10 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-02-04 12:15:00 | 4025.00 | 2026-02-05 09:15:00 | 3969.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-04 13:15:00 | 4025.20 | 2026-02-05 09:15:00 | 3969.90 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-02-12 11:45:00 | 4063.10 | 2026-02-26 09:15:00 | 4469.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 12:45:00 | 4065.00 | 2026-02-27 13:15:00 | 4356.60 | STOP_HIT | 1.00 | 7.17% |
| SELL | retest2 | 2026-03-04 09:15:00 | 4311.30 | 2026-03-05 12:15:00 | 4363.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-05 10:15:00 | 4356.10 | 2026-03-05 12:15:00 | 4363.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2026-03-12 13:00:00 | 4440.00 | 2026-03-13 10:15:00 | 4387.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-03-13 09:15:00 | 4436.80 | 2026-03-13 10:15:00 | 4387.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-03-17 15:15:00 | 4302.90 | 2026-03-25 09:15:00 | 4299.30 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2026-03-18 09:30:00 | 4302.30 | 2026-03-25 09:15:00 | 4299.30 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2026-04-01 10:15:00 | 4234.10 | 2026-04-02 09:15:00 | 4022.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 4234.10 | 2026-04-06 12:15:00 | 3999.80 | STOP_HIT | 0.50 | 5.53% |
| BUY | retest1 | 2026-04-15 09:15:00 | 4199.60 | 2026-04-15 14:15:00 | 4160.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-23 11:30:00 | 4140.70 | 2026-04-23 14:15:00 | 4146.20 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-29 09:15:00 | 4246.40 | 2026-04-30 09:15:00 | 4163.80 | STOP_HIT | 1.00 | -1.95% |
