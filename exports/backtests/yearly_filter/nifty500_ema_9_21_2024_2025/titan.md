# Titan Company Ltd. (TITAN)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 141 |
| ALERT1 | 98 |
| ALERT2 | 98 |
| ALERT2_SKIP | 51 |
| ALERT3 | 260 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 125 |
| PARTIAL | 6 |
| TARGET_HIT | 0 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 131 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 51 / 80
- **Target hits / Stop hits / Partials:** 0 / 125 / 6
- **Avg / median % per leg:** 0.37% / -0.44%
- **Sum % (uncompounded):** 48.66%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 32 | 44.4% | 0 | 72 | 0 | 0.93% | 67.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 72 | 32 | 44.4% | 0 | 72 | 0 | 0.93% | 67.0% |
| SELL (all) | 59 | 19 | 32.2% | 0 | 53 | 6 | -0.31% | -18.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 19 | 32.2% | 0 | 53 | 6 | -0.31% | -18.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 131 | 51 | 38.9% | 0 | 125 | 6 | 0.37% | 48.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 3300.35 | 3276.98 | 3276.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 3303.10 | 3285.89 | 3280.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 3276.00 | 3285.11 | 3281.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 3276.00 | 3285.11 | 3281.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 3276.00 | 3285.11 | 3281.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 3276.00 | 3285.11 | 3281.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 3272.00 | 3282.49 | 3280.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 3267.65 | 3282.49 | 3280.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 3285.70 | 3283.13 | 3281.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:30:00 | 3268.00 | 3283.13 | 3281.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 3273.45 | 3281.19 | 3280.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 13:00:00 | 3273.45 | 3281.19 | 3280.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 13:15:00 | 3272.00 | 3279.36 | 3279.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 10:15:00 | 3264.15 | 3272.33 | 3275.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 3285.55 | 3274.98 | 3276.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 3285.55 | 3274.98 | 3276.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 3285.55 | 3274.98 | 3276.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 3285.55 | 3274.98 | 3276.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 3269.45 | 3273.87 | 3276.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 13:30:00 | 3258.10 | 3271.62 | 3274.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 14:00:00 | 3262.60 | 3271.62 | 3274.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 14:15:00 | 3332.85 | 3283.86 | 3280.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 3332.85 | 3283.86 | 3280.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 15:15:00 | 3338.00 | 3294.69 | 3285.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 14:15:00 | 3382.20 | 3382.41 | 3366.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:15:00 | 3382.95 | 3382.41 | 3366.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 3411.00 | 3422.40 | 3410.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:15:00 | 3410.00 | 3422.40 | 3410.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 3397.35 | 3417.39 | 3409.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 3397.35 | 3417.39 | 3409.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 3409.20 | 3415.75 | 3409.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:30:00 | 3426.90 | 3418.71 | 3411.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:30:00 | 3418.45 | 3412.63 | 3411.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 12:15:00 | 3402.40 | 3409.22 | 3410.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 3402.40 | 3409.22 | 3410.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 3378.30 | 3401.46 | 3405.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 3313.95 | 3310.14 | 3343.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:30:00 | 3314.95 | 3310.14 | 3343.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 3321.00 | 3285.63 | 3312.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:30:00 | 3323.85 | 3285.63 | 3312.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 3274.00 | 3283.31 | 3308.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:00:00 | 3259.00 | 3278.45 | 3304.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:30:00 | 3257.55 | 3273.56 | 3299.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 3258.20 | 3270.48 | 3296.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:45:00 | 3261.95 | 3268.83 | 3292.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3226.55 | 3259.58 | 3282.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:30:00 | 3276.65 | 3259.58 | 3282.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 3096.05 | 3227.67 | 3262.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 3094.67 | 3227.67 | 3262.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 3095.29 | 3227.67 | 3262.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 3098.85 | 3227.67 | 3262.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 14:15:00 | 3254.20 | 3231.85 | 3258.57 | SL hit (close>ema200) qty=0.50 sl=3231.85 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 3329.45 | 3276.44 | 3272.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 3358.15 | 3312.21 | 3293.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 3310.10 | 3311.79 | 3295.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 3310.10 | 3311.79 | 3295.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 3306.25 | 3310.68 | 3296.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 3306.25 | 3310.68 | 3296.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 3318.95 | 3311.43 | 3299.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 3298.10 | 3311.43 | 3299.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 3366.00 | 3325.79 | 3307.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:30:00 | 3380.00 | 3344.86 | 3320.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 3382.15 | 3391.54 | 3392.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 3382.15 | 3391.54 | 3392.27 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 09:15:00 | 3419.00 | 3395.98 | 3394.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 11:15:00 | 3433.80 | 3406.12 | 3399.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 3477.30 | 3553.92 | 3522.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 3477.30 | 3553.92 | 3522.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 3477.30 | 3553.92 | 3522.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 3477.30 | 3553.92 | 3522.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 3483.65 | 3539.86 | 3519.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 3483.65 | 3539.86 | 3519.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 3461.00 | 3500.68 | 3505.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 3459.00 | 3492.35 | 3501.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 3420.05 | 3410.22 | 3430.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 12:00:00 | 3420.05 | 3410.22 | 3430.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 12:15:00 | 3375.30 | 3379.65 | 3390.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 13:15:00 | 3370.60 | 3379.65 | 3390.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 3395.00 | 3381.97 | 3387.93 | SL hit (close>static) qty=1.00 sl=3392.50 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 12:15:00 | 3401.60 | 3393.22 | 3392.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 13:15:00 | 3408.90 | 3396.36 | 3393.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 3411.75 | 3417.67 | 3408.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 3411.75 | 3417.67 | 3408.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 3411.75 | 3417.67 | 3408.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:45:00 | 3410.15 | 3417.67 | 3408.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 3396.00 | 3412.39 | 3407.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 3396.00 | 3412.39 | 3407.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 3393.90 | 3408.69 | 3406.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 3393.90 | 3408.69 | 3406.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 14:15:00 | 3398.80 | 3404.85 | 3405.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 11:15:00 | 3389.50 | 3399.63 | 3402.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 3190.95 | 3180.26 | 3231.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:30:00 | 3200.00 | 3180.26 | 3231.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 3216.90 | 3196.62 | 3220.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 15:00:00 | 3216.90 | 3196.62 | 3220.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 3235.20 | 3207.12 | 3221.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 3235.20 | 3207.12 | 3221.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 3229.90 | 3211.67 | 3222.17 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 3246.25 | 3229.25 | 3227.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 10:15:00 | 3260.35 | 3246.14 | 3238.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 12:15:00 | 3242.20 | 3247.41 | 3240.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 12:15:00 | 3242.20 | 3247.41 | 3240.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 3242.20 | 3247.41 | 3240.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:45:00 | 3247.85 | 3247.41 | 3240.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 3247.15 | 3247.36 | 3241.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 3238.55 | 3247.36 | 3241.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 3228.30 | 3243.54 | 3240.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 3228.30 | 3243.54 | 3240.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 3221.45 | 3239.13 | 3238.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 3241.40 | 3239.13 | 3238.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 09:15:00 | 3220.00 | 3235.30 | 3236.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 3220.00 | 3235.30 | 3236.80 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 11:15:00 | 3241.50 | 3232.99 | 3232.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 3248.00 | 3237.81 | 3235.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 3225.10 | 3235.27 | 3234.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 3225.10 | 3235.27 | 3234.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 3225.10 | 3235.27 | 3234.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 3225.10 | 3235.27 | 3234.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 3230.35 | 3234.28 | 3234.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:00:00 | 3230.35 | 3234.28 | 3234.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 3257.00 | 3240.88 | 3237.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 3262.45 | 3240.88 | 3237.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 11:15:00 | 3265.00 | 3248.02 | 3242.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 3260.00 | 3250.42 | 3244.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:30:00 | 3261.10 | 3255.67 | 3250.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 3269.00 | 3258.34 | 3252.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 12:00:00 | 3378.90 | 3281.01 | 3265.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 14:15:00 | 3406.10 | 3429.75 | 3430.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 3406.10 | 3429.75 | 3430.86 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 09:15:00 | 3448.25 | 3433.17 | 3432.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 10:15:00 | 3461.95 | 3438.93 | 3434.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 13:15:00 | 3457.75 | 3460.96 | 3452.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 14:00:00 | 3457.75 | 3460.96 | 3452.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 3461.40 | 3461.05 | 3452.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 15:15:00 | 3464.65 | 3461.05 | 3452.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 3450.70 | 3463.77 | 3458.13 | SL hit (close<static) qty=1.00 sl=3451.35 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 3418.95 | 3454.01 | 3455.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 3407.50 | 3438.45 | 3445.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 3323.00 | 3312.35 | 3333.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 3323.00 | 3312.35 | 3333.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 3323.00 | 3312.35 | 3333.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:45:00 | 3310.80 | 3309.26 | 3329.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 3310.25 | 3318.29 | 3326.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 10:15:00 | 3350.10 | 3329.70 | 3327.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 3350.10 | 3329.70 | 3327.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-13 11:15:00 | 3399.70 | 3343.70 | 3334.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 09:15:00 | 3386.30 | 3390.67 | 3374.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-16 10:00:00 | 3386.30 | 3390.67 | 3374.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 3384.50 | 3389.44 | 3375.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 3384.45 | 3389.44 | 3375.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 3552.30 | 3580.68 | 3548.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:30:00 | 3556.70 | 3580.68 | 3548.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 3578.15 | 3575.86 | 3555.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:15:00 | 3590.80 | 3573.74 | 3558.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 3550.00 | 3585.06 | 3584.21 | SL hit (close<static) qty=1.00 sl=3555.10 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 15:15:00 | 3560.00 | 3580.05 | 3582.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 11:15:00 | 3542.00 | 3565.15 | 3574.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 3549.80 | 3538.19 | 3551.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 12:15:00 | 3549.80 | 3538.19 | 3551.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 3549.80 | 3538.19 | 3551.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:45:00 | 3543.95 | 3538.19 | 3551.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 3519.65 | 3534.48 | 3548.77 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 3568.00 | 3555.82 | 3554.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 11:15:00 | 3577.50 | 3564.83 | 3559.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 09:15:00 | 3565.35 | 3575.64 | 3568.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 3565.35 | 3575.64 | 3568.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 3565.35 | 3575.64 | 3568.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 10:00:00 | 3565.35 | 3575.64 | 3568.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 3581.15 | 3576.74 | 3569.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:15:00 | 3590.40 | 3576.74 | 3569.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 3590.10 | 3600.58 | 3588.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:00:00 | 3598.75 | 3600.21 | 3589.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 3729.40 | 3746.81 | 3747.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 3729.40 | 3746.81 | 3747.32 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 11:15:00 | 3753.95 | 3748.27 | 3747.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 12:15:00 | 3761.70 | 3750.95 | 3749.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 11:15:00 | 3759.70 | 3762.86 | 3757.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 11:15:00 | 3759.70 | 3762.86 | 3757.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 3759.70 | 3762.86 | 3757.29 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 3724.00 | 3749.43 | 3751.80 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 10:15:00 | 3774.95 | 3753.81 | 3752.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 3794.00 | 3770.08 | 3761.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 13:15:00 | 3784.10 | 3787.36 | 3774.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 13:15:00 | 3784.10 | 3787.36 | 3774.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 13:15:00 | 3784.10 | 3787.36 | 3774.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 13:30:00 | 3766.90 | 3787.36 | 3774.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 3799.20 | 3789.73 | 3776.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 09:30:00 | 3815.20 | 3796.72 | 3782.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 12:30:00 | 3812.95 | 3803.89 | 3789.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 09:45:00 | 3815.30 | 3811.19 | 3798.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 12:00:00 | 3812.00 | 3812.44 | 3801.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 3783.70 | 3806.84 | 3801.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 3783.70 | 3806.84 | 3801.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 3785.00 | 3802.47 | 3799.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 3781.15 | 3802.47 | 3799.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 3762.60 | 3794.50 | 3796.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 3762.60 | 3794.50 | 3796.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 10:15:00 | 3758.70 | 3787.34 | 3793.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 3756.35 | 3737.12 | 3753.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 3756.35 | 3737.12 | 3753.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 3756.35 | 3737.12 | 3753.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 3756.35 | 3737.12 | 3753.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 3754.00 | 3740.50 | 3753.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 3793.70 | 3740.50 | 3753.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 3832.30 | 3758.86 | 3760.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:45:00 | 3827.30 | 3758.86 | 3760.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 3831.65 | 3773.42 | 3766.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 11:15:00 | 3856.05 | 3789.94 | 3774.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 14:15:00 | 3820.65 | 3823.08 | 3806.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 15:00:00 | 3820.65 | 3823.08 | 3806.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3781.60 | 3815.57 | 3806.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 3781.60 | 3815.57 | 3806.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 3782.55 | 3808.97 | 3804.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 3782.00 | 3808.97 | 3804.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 3760.00 | 3795.28 | 3798.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 3750.45 | 3778.93 | 3788.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 10:15:00 | 3724.90 | 3714.20 | 3742.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:45:00 | 3728.90 | 3714.20 | 3742.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 3742.40 | 3719.84 | 3742.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:00:00 | 3742.40 | 3719.84 | 3742.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 3686.85 | 3713.24 | 3737.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 12:30:00 | 3707.15 | 3713.24 | 3737.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 3582.00 | 3676.26 | 3711.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 3546.50 | 3608.06 | 3656.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 3545.20 | 3595.49 | 3645.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:30:00 | 3545.00 | 3583.13 | 3635.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:30:00 | 3538.40 | 3527.40 | 3572.01 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 3499.40 | 3475.73 | 3489.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 3499.40 | 3475.73 | 3489.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 3514.45 | 3483.47 | 3491.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:45:00 | 3509.95 | 3483.47 | 3491.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 3502.15 | 3492.57 | 3494.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:00:00 | 3502.15 | 3492.57 | 3494.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 3495.40 | 3493.13 | 3494.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 15:15:00 | 3508.00 | 3493.13 | 3494.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-14 15:15:00 | 3508.00 | 3496.11 | 3495.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 15:15:00 | 3508.00 | 3496.11 | 3495.75 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 09:15:00 | 3492.10 | 3495.30 | 3495.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 11:15:00 | 3484.70 | 3492.07 | 3493.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 13:15:00 | 3507.45 | 3493.70 | 3494.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 13:15:00 | 3507.45 | 3493.70 | 3494.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 3507.45 | 3493.70 | 3494.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:00:00 | 3507.45 | 3493.70 | 3494.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 3508.75 | 3496.71 | 3495.52 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 3467.45 | 3490.20 | 3492.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 3453.50 | 3482.86 | 3489.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 3474.50 | 3472.79 | 3482.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-16 15:00:00 | 3474.50 | 3472.79 | 3482.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 3447.05 | 3466.56 | 3477.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 11:30:00 | 3434.30 | 3457.25 | 3471.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 3262.59 | 3306.61 | 3320.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 09:15:00 | 3283.05 | 3276.94 | 3296.10 | SL hit (close>ema200) qty=0.50 sl=3276.94 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 3311.05 | 3289.09 | 3286.90 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 10:15:00 | 3265.00 | 3286.78 | 3287.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 3263.70 | 3278.56 | 3283.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 15:15:00 | 3277.00 | 3274.51 | 3280.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 3301.95 | 3280.00 | 3282.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 3301.95 | 3280.00 | 3282.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 3301.95 | 3280.00 | 3282.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-01 18:15:00 | 3300.00 | 3284.00 | 3283.67 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 3215.55 | 3270.31 | 3277.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 13:15:00 | 3203.40 | 3239.52 | 3258.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 3243.70 | 3235.02 | 3251.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 3243.70 | 3235.02 | 3251.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 3243.70 | 3235.02 | 3251.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 3243.70 | 3235.02 | 3251.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 3235.00 | 3221.05 | 3235.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:15:00 | 3129.35 | 3221.05 | 3235.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 3217.50 | 3178.68 | 3173.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 10:15:00 | 3217.50 | 3178.68 | 3173.86 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 09:15:00 | 3168.60 | 3188.46 | 3190.92 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 3220.05 | 3182.74 | 3180.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 3258.25 | 3197.84 | 3187.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 3222.35 | 3225.65 | 3206.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 3222.35 | 3225.65 | 3206.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 3205.00 | 3220.61 | 3207.47 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 3166.10 | 3196.09 | 3198.24 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 10:15:00 | 3220.50 | 3197.87 | 3197.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 11:15:00 | 3249.95 | 3208.29 | 3201.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 3304.00 | 3316.16 | 3296.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 3300.90 | 3313.11 | 3297.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 3300.90 | 3313.11 | 3297.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 3298.45 | 3313.11 | 3297.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 3281.70 | 3306.82 | 3295.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 3281.70 | 3306.82 | 3295.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 3303.75 | 3306.21 | 3296.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:15:00 | 3308.25 | 3306.21 | 3296.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 3251.80 | 3288.75 | 3291.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 3251.80 | 3288.75 | 3291.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 3227.30 | 3276.46 | 3285.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 3255.10 | 3241.22 | 3259.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 3255.10 | 3241.22 | 3259.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 3251.50 | 3244.11 | 3256.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 13:30:00 | 3251.15 | 3244.11 | 3256.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 3249.00 | 3245.09 | 3255.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 3253.90 | 3245.09 | 3255.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 3245.00 | 3245.07 | 3254.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 3236.10 | 3245.07 | 3254.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 3261.85 | 3248.43 | 3255.35 | SL hit (close>static) qty=1.00 sl=3255.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 3282.40 | 3261.10 | 3259.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 3300.00 | 3268.88 | 3263.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 3453.90 | 3459.36 | 3425.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 09:15:00 | 3485.00 | 3469.02 | 3447.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 3485.00 | 3469.02 | 3447.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 10:30:00 | 3508.65 | 3476.44 | 3452.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 09:15:00 | 3434.45 | 3465.56 | 3464.74 | SL hit (close<static) qty=1.00 sl=3436.05 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 3446.10 | 3461.67 | 3463.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 3409.40 | 3441.46 | 3451.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 3459.45 | 3436.98 | 3447.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 3459.45 | 3436.98 | 3447.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 3459.45 | 3436.98 | 3447.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 3459.45 | 3436.98 | 3447.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 3468.95 | 3443.38 | 3449.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 3468.95 | 3443.38 | 3449.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 3491.50 | 3453.00 | 3452.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 3511.40 | 3464.68 | 3458.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 3468.45 | 3472.85 | 3463.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 3468.45 | 3472.85 | 3463.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 3468.45 | 3472.85 | 3463.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 3477.55 | 3472.85 | 3463.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 3448.55 | 3467.99 | 3462.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 3448.55 | 3467.99 | 3462.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 3430.80 | 3460.55 | 3459.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:00:00 | 3430.80 | 3460.55 | 3459.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 12:15:00 | 3439.90 | 3456.42 | 3457.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 09:15:00 | 3425.25 | 3443.34 | 3450.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 11:15:00 | 3413.75 | 3409.69 | 3424.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:30:00 | 3406.20 | 3409.69 | 3424.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 3386.40 | 3373.45 | 3390.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:30:00 | 3416.20 | 3373.45 | 3390.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 3406.80 | 3380.12 | 3391.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 11:15:00 | 3380.95 | 3380.12 | 3391.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 14:15:00 | 3397.50 | 3383.58 | 3382.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 14:15:00 | 3397.50 | 3383.58 | 3382.90 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 12:15:00 | 3356.00 | 3381.07 | 3382.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 3349.90 | 3365.20 | 3373.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 3342.00 | 3338.92 | 3352.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 10:30:00 | 3341.55 | 3338.92 | 3352.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 3261.15 | 3257.02 | 3267.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 10:00:00 | 3261.15 | 3257.02 | 3267.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 3278.25 | 3261.27 | 3268.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 11:00:00 | 3278.25 | 3261.27 | 3268.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 3297.05 | 3268.42 | 3271.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:00:00 | 3297.05 | 3268.42 | 3271.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 3337.50 | 3282.24 | 3277.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 3362.25 | 3298.24 | 3285.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 3467.00 | 3467.03 | 3422.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 14:30:00 | 3469.30 | 3467.03 | 3422.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 3442.65 | 3490.05 | 3467.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 3442.65 | 3490.05 | 3467.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 3431.20 | 3478.28 | 3464.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 11:00:00 | 3431.20 | 3478.28 | 3464.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 3476.85 | 3473.61 | 3465.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 3464.15 | 3473.61 | 3465.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 12:15:00 | 3487.40 | 3481.90 | 3472.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 12:30:00 | 3485.05 | 3481.90 | 3472.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 3438.90 | 3475.94 | 3472.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:45:00 | 3430.85 | 3475.94 | 3472.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 3463.55 | 3473.47 | 3472.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 3478.00 | 3473.47 | 3472.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 12:15:00 | 3456.80 | 3468.56 | 3469.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 3456.80 | 3468.56 | 3469.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 3443.75 | 3463.60 | 3467.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 3324.80 | 3319.78 | 3351.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 15:00:00 | 3324.80 | 3319.78 | 3351.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 3331.80 | 3313.35 | 3327.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 3331.80 | 3313.35 | 3327.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 3321.70 | 3315.02 | 3326.58 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 13:15:00 | 3353.90 | 3332.51 | 3332.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 09:15:00 | 3394.00 | 3352.01 | 3342.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 3373.65 | 3378.98 | 3364.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 09:45:00 | 3375.85 | 3378.98 | 3364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3349.00 | 3372.98 | 3362.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:15:00 | 3341.00 | 3372.98 | 3362.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 3381.05 | 3374.59 | 3364.44 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 3344.10 | 3361.99 | 3363.67 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 3392.60 | 3367.29 | 3365.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 11:15:00 | 3409.10 | 3375.65 | 3369.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 3391.20 | 3391.41 | 3381.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 3391.20 | 3391.41 | 3381.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3391.20 | 3391.41 | 3381.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 3391.20 | 3391.41 | 3381.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 3410.90 | 3395.31 | 3383.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:15:00 | 3414.85 | 3395.31 | 3383.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 3370.00 | 3395.77 | 3390.76 | SL hit (close<static) qty=1.00 sl=3383.70 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 3368.75 | 3385.04 | 3386.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 3361.15 | 3380.27 | 3384.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 3349.70 | 3342.40 | 3358.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 3349.70 | 3342.40 | 3358.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 3348.70 | 3345.51 | 3357.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:15:00 | 3357.00 | 3345.51 | 3357.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 3329.45 | 3342.30 | 3355.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 3361.90 | 3342.30 | 3355.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 3354.00 | 3342.21 | 3352.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 3354.00 | 3342.21 | 3352.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 3349.35 | 3343.64 | 3352.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 3365.90 | 3343.64 | 3352.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 3359.10 | 3346.73 | 3352.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 3354.45 | 3346.73 | 3352.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 3356.70 | 3348.72 | 3353.30 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 3372.05 | 3356.81 | 3356.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 3375.95 | 3362.85 | 3359.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 3358.55 | 3366.08 | 3362.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 3358.55 | 3366.08 | 3362.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 3358.55 | 3366.08 | 3362.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 3358.55 | 3366.08 | 3362.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 3369.20 | 3366.70 | 3362.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 15:15:00 | 3371.65 | 3366.70 | 3362.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 11:15:00 | 3500.30 | 3535.27 | 3536.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 3500.30 | 3535.27 | 3536.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 3435.05 | 3496.33 | 3515.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 3438.10 | 3437.01 | 3469.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 10:00:00 | 3438.10 | 3437.01 | 3469.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 3236.40 | 3213.84 | 3229.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 3236.40 | 3213.84 | 3229.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 3226.45 | 3216.36 | 3229.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 3218.35 | 3216.36 | 3229.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 3220.10 | 3217.11 | 3228.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 3202.40 | 3217.11 | 3228.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 3244.35 | 3223.10 | 3225.48 | SL hit (close>static) qty=1.00 sl=3244.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 12:15:00 | 3199.45 | 3189.77 | 3189.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 14:15:00 | 3225.00 | 3200.66 | 3195.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 09:15:00 | 3151.65 | 3194.75 | 3193.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 09:15:00 | 3151.65 | 3194.75 | 3193.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 3151.65 | 3194.75 | 3193.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 10:00:00 | 3151.65 | 3194.75 | 3193.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 10:15:00 | 3115.85 | 3178.97 | 3186.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 11:15:00 | 3085.20 | 3160.22 | 3177.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 3048.95 | 3048.61 | 3076.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:15:00 | 3064.85 | 3048.61 | 3076.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 3067.30 | 3052.35 | 3076.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 3072.80 | 3052.35 | 3076.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 10:15:00 | 3088.30 | 3059.54 | 3077.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:00:00 | 3088.30 | 3059.54 | 3077.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 11:15:00 | 3090.50 | 3065.73 | 3078.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 11:45:00 | 3092.60 | 3065.73 | 3078.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 3086.30 | 3078.77 | 3081.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:00:00 | 3086.30 | 3078.77 | 3081.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 3084.00 | 3079.81 | 3081.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 3084.00 | 3079.81 | 3081.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 3078.10 | 3079.47 | 3081.13 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 14:15:00 | 3118.35 | 3088.03 | 3084.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 3125.00 | 3095.42 | 3088.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 14:15:00 | 3086.60 | 3101.28 | 3095.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 14:15:00 | 3086.60 | 3101.28 | 3095.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 3086.60 | 3101.28 | 3095.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 3086.60 | 3101.28 | 3095.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 3080.00 | 3097.02 | 3094.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 3052.25 | 3097.02 | 3094.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 3048.95 | 3087.41 | 3090.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 3035.50 | 3072.16 | 3082.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 3051.00 | 3048.38 | 3064.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 10:00:00 | 3051.00 | 3048.38 | 3064.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 3040.10 | 3027.68 | 3038.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 3040.10 | 3027.68 | 3038.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 3041.00 | 3030.34 | 3039.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:30:00 | 3019.10 | 3027.83 | 3037.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 3050.55 | 3022.60 | 3024.19 | SL hit (close>static) qty=1.00 sl=3049.50 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 3050.45 | 3028.17 | 3026.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 11:15:00 | 3061.45 | 3034.83 | 3029.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 11:15:00 | 3063.85 | 3065.77 | 3051.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 11:30:00 | 3067.50 | 3065.77 | 3051.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 3092.10 | 3152.40 | 3138.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 3098.00 | 3152.40 | 3138.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 3092.10 | 3140.34 | 3134.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:15:00 | 3087.90 | 3140.34 | 3134.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 11:15:00 | 3081.50 | 3128.57 | 3129.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 15:15:00 | 3076.00 | 3100.41 | 3114.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 3069.35 | 3066.98 | 3085.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 10:00:00 | 3069.35 | 3066.98 | 3085.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 3083.60 | 3067.86 | 3076.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 3085.80 | 3067.86 | 3076.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 3079.70 | 3070.23 | 3076.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 3082.40 | 3070.23 | 3076.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 3076.70 | 3071.52 | 3076.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:45:00 | 3078.30 | 3071.52 | 3076.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 3070.95 | 3071.41 | 3076.39 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 3096.95 | 3080.19 | 3079.22 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 3060.55 | 3077.33 | 3078.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 3012.80 | 3058.95 | 3069.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 3028.40 | 3012.57 | 3034.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 3028.40 | 3012.57 | 3034.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 3040.65 | 3018.19 | 3035.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:45:00 | 3045.40 | 3018.19 | 3035.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 3085.00 | 3031.55 | 3039.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 3085.00 | 3031.55 | 3039.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 3101.10 | 3045.46 | 3045.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 3144.40 | 3081.28 | 3063.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 3088.00 | 3111.26 | 3092.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 3088.00 | 3111.26 | 3092.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 3088.00 | 3111.26 | 3092.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 3088.00 | 3111.26 | 3092.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 3098.20 | 3108.65 | 3092.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:45:00 | 3093.95 | 3108.65 | 3092.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 3085.20 | 3103.96 | 3091.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 11:45:00 | 3085.70 | 3103.96 | 3091.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 3062.30 | 3095.63 | 3089.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:00:00 | 3062.30 | 3095.63 | 3089.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 3078.00 | 3083.96 | 3084.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 2982.65 | 3063.70 | 3075.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 3046.10 | 3029.39 | 3048.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 3189.00 | 3029.39 | 3048.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3050.20 | 3056.45 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 10:15:00 | 3132.00 | 3066.56 | 3063.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 11:15:00 | 3150.50 | 3083.35 | 3071.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 3256.90 | 3260.27 | 3229.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 3256.90 | 3260.27 | 3229.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3257.50 | 3265.57 | 3246.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:15:00 | 3270.50 | 3265.57 | 3246.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:15:00 | 3281.90 | 3265.26 | 3247.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 10:15:00 | 3343.90 | 3373.98 | 3375.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 10:15:00 | 3343.90 | 3373.98 | 3375.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 15:15:00 | 3330.00 | 3353.66 | 3363.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 3372.70 | 3357.47 | 3364.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 3372.70 | 3357.47 | 3364.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 3372.70 | 3357.47 | 3364.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:00:00 | 3344.90 | 3354.96 | 3362.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:30:00 | 3346.10 | 3353.46 | 3361.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 12:30:00 | 3346.00 | 3352.19 | 3360.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:00:00 | 3347.10 | 3352.19 | 3360.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 3348.00 | 3307.37 | 3318.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 3348.00 | 3307.37 | 3318.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 3357.00 | 3317.30 | 3322.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 3365.50 | 3317.30 | 3322.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-07 13:15:00 | 3348.80 | 3329.38 | 3327.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 13:15:00 | 3348.80 | 3329.38 | 3327.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 3361.70 | 3339.32 | 3332.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 3345.00 | 3363.23 | 3351.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 15:15:00 | 3345.00 | 3363.23 | 3351.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 3345.00 | 3363.23 | 3351.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 3516.10 | 3363.23 | 3351.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 3597.90 | 3611.60 | 3612.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 3597.90 | 3611.60 | 3612.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 3581.20 | 3605.52 | 3609.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 3608.60 | 3601.96 | 3606.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 3608.60 | 3601.96 | 3606.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 3613.00 | 3604.16 | 3607.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:45:00 | 3616.00 | 3604.16 | 3607.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 3580.00 | 3599.33 | 3604.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 3577.70 | 3599.33 | 3604.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 3578.30 | 3594.66 | 3601.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 3576.80 | 3591.73 | 3599.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 3537.10 | 3590.18 | 3598.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3573.10 | 3558.09 | 3572.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 3573.10 | 3558.09 | 3572.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 3597.40 | 3565.95 | 3574.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:00:00 | 3597.40 | 3565.95 | 3574.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 3585.00 | 3569.76 | 3575.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:30:00 | 3579.40 | 3576.85 | 3577.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 3628.90 | 3587.76 | 3582.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 3628.90 | 3587.76 | 3582.60 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 3563.00 | 3585.82 | 3588.63 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 3600.00 | 3585.41 | 3583.75 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 3566.00 | 3580.52 | 3581.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 14:15:00 | 3549.10 | 3571.88 | 3577.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 15:15:00 | 3534.90 | 3530.50 | 3547.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:15:00 | 3519.80 | 3530.50 | 3547.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 3520.00 | 3528.40 | 3544.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 09:15:00 | 3498.00 | 3522.62 | 3534.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 3505.00 | 3508.10 | 3516.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:00:00 | 3503.60 | 3510.83 | 3516.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 12:15:00 | 3537.40 | 3517.08 | 3516.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 3537.40 | 3517.08 | 3516.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 3554.30 | 3524.53 | 3519.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 3531.90 | 3538.42 | 3529.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:30:00 | 3535.30 | 3538.42 | 3529.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 3530.90 | 3536.92 | 3529.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:45:00 | 3526.90 | 3536.92 | 3529.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 3536.10 | 3536.75 | 3529.93 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 3520.70 | 3528.50 | 3528.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 3512.50 | 3523.17 | 3525.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3532.80 | 3522.86 | 3525.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 3532.80 | 3522.86 | 3525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 3527.10 | 3523.71 | 3525.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:15:00 | 3520.30 | 3523.71 | 3525.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 14:15:00 | 3539.70 | 3527.00 | 3526.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 3539.70 | 3527.00 | 3526.59 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 3502.10 | 3523.79 | 3525.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 3499.00 | 3518.83 | 3523.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 3433.70 | 3431.21 | 3452.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:30:00 | 3434.60 | 3431.21 | 3452.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 3436.90 | 3434.22 | 3450.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 3442.80 | 3434.22 | 3450.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3428.60 | 3417.00 | 3429.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 3431.50 | 3417.00 | 3429.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 3418.40 | 3417.28 | 3428.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:15:00 | 3411.20 | 3417.28 | 3428.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 14:15:00 | 3476.90 | 3432.82 | 3432.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 3476.90 | 3432.82 | 3432.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 15:15:00 | 3478.80 | 3442.02 | 3436.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 15:15:00 | 3510.40 | 3513.97 | 3494.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:15:00 | 3493.50 | 3513.97 | 3494.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3475.80 | 3506.34 | 3492.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 3475.80 | 3506.34 | 3492.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3487.70 | 3502.61 | 3491.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:00:00 | 3489.00 | 3499.89 | 3491.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 14:15:00 | 3680.90 | 3692.89 | 3693.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 14:15:00 | 3680.90 | 3692.89 | 3693.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 3663.80 | 3685.17 | 3689.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 12:15:00 | 3677.50 | 3675.32 | 3683.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 13:00:00 | 3677.50 | 3675.32 | 3683.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3687.80 | 3677.57 | 3682.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 3687.80 | 3677.57 | 3682.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 3684.00 | 3678.86 | 3682.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 3691.20 | 3678.19 | 3682.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 3404.90 | 3388.43 | 3407.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 3410.00 | 3388.43 | 3407.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3402.00 | 3391.14 | 3406.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3413.00 | 3391.14 | 3406.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3395.00 | 3391.91 | 3405.63 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 3415.40 | 3409.60 | 3408.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 10:15:00 | 3436.00 | 3417.33 | 3412.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 3429.90 | 3429.96 | 3421.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 3424.50 | 3429.96 | 3421.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 3414.40 | 3426.85 | 3421.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:45:00 | 3413.80 | 3426.85 | 3421.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3406.30 | 3422.74 | 3419.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 3403.70 | 3422.74 | 3419.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 3407.60 | 3417.16 | 3417.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3404.40 | 3413.11 | 3415.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 3428.00 | 3408.19 | 3411.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 13:00:00 | 3428.00 | 3408.19 | 3411.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 3433.80 | 3413.31 | 3413.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 3470.20 | 3430.19 | 3421.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3460.00 | 3461.97 | 3446.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:30:00 | 3450.20 | 3461.97 | 3446.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 3472.80 | 3470.11 | 3458.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 10:15:00 | 3481.10 | 3470.11 | 3458.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 3477.90 | 3479.12 | 3469.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 11:15:00 | 3456.50 | 3471.69 | 3468.06 | SL hit (close<static) qty=1.00 sl=3458.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 15:15:00 | 3451.20 | 3466.69 | 3466.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 09:15:00 | 3433.60 | 3460.07 | 3463.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 3377.60 | 3369.46 | 3387.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 15:00:00 | 3377.60 | 3369.46 | 3387.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 3338.00 | 3330.53 | 3345.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:30:00 | 3333.70 | 3330.53 | 3345.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 3351.00 | 3335.86 | 3344.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 3351.00 | 3335.86 | 3344.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 3359.00 | 3340.48 | 3345.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 3359.00 | 3340.48 | 3345.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3358.70 | 3344.13 | 3346.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 3355.90 | 3344.13 | 3346.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 10:15:00 | 3358.00 | 3349.18 | 3348.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 11:15:00 | 3374.00 | 3354.15 | 3350.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 14:15:00 | 3418.90 | 3422.24 | 3399.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 15:00:00 | 3418.90 | 3422.24 | 3399.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3426.50 | 3419.53 | 3402.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 3442.90 | 3408.56 | 3403.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:45:00 | 3448.30 | 3420.73 | 3409.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 3437.40 | 3447.22 | 3434.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 3591.60 | 3621.11 | 3621.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 3591.60 | 3621.11 | 3621.12 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 3641.00 | 3623.07 | 3621.42 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 3613.00 | 3620.54 | 3621.55 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3634.50 | 3623.98 | 3622.68 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 15:15:00 | 3618.00 | 3621.71 | 3622.10 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 3637.00 | 3624.77 | 3623.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 3642.50 | 3628.32 | 3625.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3617.60 | 3629.49 | 3626.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 3617.60 | 3629.49 | 3626.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3623.00 | 3628.19 | 3626.44 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 3612.20 | 3624.99 | 3625.15 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 3630.80 | 3626.15 | 3625.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 3648.30 | 3630.58 | 3627.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 3678.00 | 3683.97 | 3669.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 09:15:00 | 3681.00 | 3683.97 | 3669.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 3672.00 | 3683.04 | 3672.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:00:00 | 3672.00 | 3683.04 | 3672.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 3673.40 | 3681.11 | 3672.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 3673.40 | 3681.11 | 3672.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 3678.80 | 3680.65 | 3673.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 3673.00 | 3680.65 | 3673.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 3666.80 | 3677.88 | 3672.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 3666.80 | 3677.88 | 3672.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 3660.00 | 3674.30 | 3671.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 3655.10 | 3674.30 | 3671.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 3669.70 | 3673.38 | 3671.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 3677.30 | 3673.71 | 3671.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 3675.80 | 3673.71 | 3671.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 12:15:00 | 3659.60 | 3669.54 | 3670.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 3659.60 | 3669.54 | 3670.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 3625.00 | 3659.29 | 3665.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 12:15:00 | 3538.90 | 3533.51 | 3553.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 3538.90 | 3533.51 | 3553.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 3557.00 | 3538.21 | 3553.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:00:00 | 3557.00 | 3538.21 | 3553.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 3555.90 | 3541.74 | 3554.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3540.00 | 3544.26 | 3553.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3363.00 | 3390.38 | 3409.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-26 10:15:00 | 3393.50 | 3391.00 | 3407.77 | SL hit (close>ema200) qty=0.50 sl=3391.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 09:15:00 | 3398.10 | 3389.26 | 3388.55 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 12:15:00 | 3367.70 | 3387.58 | 3388.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3375.96 | 3381.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3384.30 | 3376.12 | 3380.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:45:00 | 3385.20 | 3376.12 | 3380.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 3385.40 | 3377.97 | 3381.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 3385.40 | 3377.97 | 3381.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 3387.60 | 3379.90 | 3381.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 3387.60 | 3379.90 | 3381.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 14:15:00 | 3406.20 | 3385.16 | 3384.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 11:15:00 | 3416.00 | 3396.98 | 3390.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3420.40 | 3421.17 | 3406.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:30:00 | 3427.00 | 3421.17 | 3406.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3407.00 | 3418.33 | 3406.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:00:00 | 3407.00 | 3418.33 | 3406.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 3425.00 | 3419.67 | 3408.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 12:15:00 | 3432.00 | 3419.67 | 3408.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:45:00 | 3430.10 | 3428.24 | 3418.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 3551.40 | 3427.88 | 3422.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 3517.00 | 3527.67 | 3528.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 3517.00 | 3527.67 | 3528.60 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 3546.00 | 3529.75 | 3528.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 3636.30 | 3557.27 | 3542.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 3742.90 | 3756.06 | 3727.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:45:00 | 3748.90 | 3756.06 | 3727.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 3724.00 | 3749.65 | 3727.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:00:00 | 3724.00 | 3749.65 | 3727.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 3732.00 | 3746.12 | 3727.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 3723.70 | 3746.12 | 3727.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 3720.00 | 3740.90 | 3727.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 3720.00 | 3740.90 | 3727.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 3714.40 | 3735.60 | 3726.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:00:00 | 3714.40 | 3735.60 | 3726.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3724.00 | 3728.76 | 3725.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 3723.50 | 3728.76 | 3725.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3738.60 | 3730.72 | 3726.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:15:00 | 3746.20 | 3730.72 | 3726.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:45:00 | 3749.50 | 3733.20 | 3727.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 14:15:00 | 3752.30 | 3734.56 | 3729.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 3760.00 | 3735.01 | 3730.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3729.90 | 3733.98 | 3730.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 3725.00 | 3733.98 | 3730.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 3692.10 | 3725.61 | 3726.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 3692.10 | 3725.61 | 3726.75 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 3738.50 | 3723.08 | 3722.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 3750.70 | 3732.26 | 3727.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3730.50 | 3737.34 | 3731.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 3725.30 | 3737.34 | 3731.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3772.00 | 3744.27 | 3735.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 3777.30 | 3753.50 | 3744.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 3675.30 | 3737.81 | 3742.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3675.30 | 3737.81 | 3742.30 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 3807.00 | 3746.37 | 3742.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 11:15:00 | 3815.40 | 3770.12 | 3754.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3791.60 | 3795.38 | 3775.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 10:15:00 | 3786.80 | 3793.66 | 3776.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3786.80 | 3793.66 | 3776.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 3780.00 | 3793.66 | 3776.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 3780.90 | 3791.11 | 3776.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 11:45:00 | 3778.70 | 3791.11 | 3776.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 3791.80 | 3791.25 | 3778.03 | EMA400 retest candle locked (from upside) |

### Cycle 102 — SELL (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 12:15:00 | 3771.00 | 3774.00 | 3774.15 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3797.00 | 3776.88 | 3775.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3831.20 | 3802.97 | 3793.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 15:15:00 | 3834.00 | 3842.70 | 3821.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:15:00 | 3842.50 | 3842.70 | 3821.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 3820.00 | 3839.12 | 3828.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 3820.00 | 3839.12 | 3828.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 3835.00 | 3838.29 | 3829.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:30:00 | 3850.00 | 3835.20 | 3829.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 11:15:00 | 3813.20 | 3829.89 | 3827.90 | SL hit (close<static) qty=1.00 sl=3817.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 3809.50 | 3825.81 | 3826.23 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 3841.70 | 3826.33 | 3825.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 11:15:00 | 3864.40 | 3836.90 | 3831.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3852.80 | 3854.47 | 3843.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:30:00 | 3848.00 | 3854.47 | 3843.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3859.00 | 3854.88 | 3846.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 3846.30 | 3854.88 | 3846.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 3902.00 | 3917.84 | 3901.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 3902.00 | 3917.84 | 3901.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 3900.00 | 3914.27 | 3901.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 3905.50 | 3914.27 | 3901.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3910.60 | 3913.54 | 3902.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:45:00 | 3925.30 | 3912.21 | 3906.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 12:15:00 | 3887.40 | 3902.78 | 3903.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 12:15:00 | 3887.40 | 3902.78 | 3903.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 3872.20 | 3894.08 | 3898.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3906.70 | 3896.30 | 3899.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3906.70 | 3896.30 | 3899.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3896.80 | 3896.40 | 3898.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 11:15:00 | 3890.70 | 3896.40 | 3898.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 3910.60 | 3882.72 | 3887.38 | SL hit (close>static) qty=1.00 sl=3908.40 alert=retest2 |

### Cycle 107 — BUY (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 14:15:00 | 3902.70 | 3892.16 | 3890.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 3915.90 | 3898.04 | 3893.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 3898.90 | 3903.78 | 3898.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:00:00 | 3898.90 | 3903.78 | 3898.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 3906.80 | 3904.39 | 3898.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 3915.90 | 3904.74 | 3899.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 3912.10 | 3910.49 | 3904.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 3884.50 | 3903.51 | 3903.29 | SL hit (close<static) qty=1.00 sl=3892.60 alert=retest2 |

### Cycle 108 — SELL (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 10:15:00 | 3866.60 | 3896.13 | 3899.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 3858.00 | 3880.70 | 3886.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 3814.10 | 3811.63 | 3831.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 10:00:00 | 3814.10 | 3811.63 | 3831.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 3808.30 | 3809.98 | 3820.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:30:00 | 3820.00 | 3809.98 | 3820.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3820.00 | 3795.44 | 3806.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 3820.00 | 3795.44 | 3806.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 3841.70 | 3804.69 | 3809.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 3841.70 | 3804.69 | 3809.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 3867.60 | 3817.27 | 3814.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 10:15:00 | 3885.30 | 3851.53 | 3840.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3864.10 | 3869.95 | 3856.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 3854.00 | 3869.95 | 3856.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 3863.20 | 3868.60 | 3857.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 3863.20 | 3868.60 | 3857.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3861.00 | 3867.08 | 3857.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 3857.10 | 3867.08 | 3857.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 15:15:00 | 3855.20 | 3863.97 | 3859.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:15:00 | 3866.80 | 3863.97 | 3859.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3887.50 | 3868.68 | 3861.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:45:00 | 3925.30 | 3899.24 | 3880.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 3921.00 | 3907.11 | 3898.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:15:00 | 3918.50 | 3907.11 | 3898.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 3919.20 | 3909.53 | 3900.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3922.70 | 3911.84 | 3903.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 3924.40 | 3912.19 | 3904.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 3924.40 | 3930.54 | 3929.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 3916.70 | 3927.77 | 3928.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 3916.70 | 3927.77 | 3928.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 3908.00 | 3923.82 | 3926.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 15:15:00 | 3920.00 | 3916.76 | 3921.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 09:15:00 | 3928.90 | 3916.76 | 3921.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 111 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 3964.30 | 3926.27 | 3925.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 3969.00 | 3941.16 | 3932.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 14:15:00 | 3981.90 | 3993.00 | 3974.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 3981.90 | 3993.00 | 3974.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 3974.80 | 3988.08 | 3975.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:45:00 | 3985.00 | 3987.30 | 3976.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 12:00:00 | 3985.10 | 3986.86 | 3976.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 3983.70 | 3986.23 | 3977.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 4008.90 | 3981.20 | 3977.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4037.50 | 3992.46 | 3982.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 4057.50 | 3992.46 | 3982.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 4056.50 | 4005.89 | 3989.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 4055.00 | 4031.94 | 4008.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 4053.50 | 4035.15 | 4012.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 4025.90 | 4044.69 | 4031.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 4060.60 | 4048.94 | 4040.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 4073.70 | 4048.94 | 4040.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 4183.20 | 4213.78 | 4215.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 4183.20 | 4213.78 | 4215.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 4164.80 | 4203.98 | 4210.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 4215.00 | 4196.16 | 4204.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 4215.00 | 4196.16 | 4204.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 4228.00 | 4202.53 | 4206.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 4238.90 | 4202.53 | 4206.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2026-01-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 09:15:00 | 4243.00 | 4214.79 | 4211.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 10:15:00 | 4259.00 | 4236.59 | 4226.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 4231.90 | 4240.55 | 4231.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 4231.90 | 4240.55 | 4231.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 4222.00 | 4236.84 | 4230.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 4220.30 | 4236.84 | 4230.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 4217.60 | 4232.99 | 4229.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 4220.00 | 4232.99 | 4229.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 4217.00 | 4226.42 | 4226.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 4194.40 | 4217.57 | 4222.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 15:15:00 | 4089.00 | 4086.24 | 4111.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:15:00 | 4116.70 | 4086.24 | 4111.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 4105.00 | 4089.99 | 4111.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 4115.70 | 4089.99 | 4111.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 4055.90 | 4042.86 | 4063.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:45:00 | 4050.40 | 4042.86 | 4063.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3974.90 | 3997.60 | 4021.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 3916.20 | 3981.01 | 4001.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 14:00:00 | 3964.80 | 3948.91 | 3956.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 3967.50 | 3959.08 | 3960.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 4044.90 | 3958.28 | 3957.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 4044.90 | 3958.28 | 3957.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 4087.30 | 3984.09 | 3969.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 3944.00 | 3978.41 | 3969.47 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3905.00 | 3955.21 | 3959.95 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 4069.00 | 3972.71 | 3964.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 4151.80 | 4080.62 | 4034.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 4092.30 | 4114.92 | 4076.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 4092.00 | 4114.92 | 4076.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 4078.80 | 4107.70 | 4076.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 4078.80 | 4107.70 | 4076.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 4066.60 | 4099.48 | 4075.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:00:00 | 4066.60 | 4099.48 | 4075.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 4082.80 | 4096.14 | 4076.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 4091.00 | 4095.21 | 4077.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 4087.40 | 4090.89 | 4081.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 13:15:00 | 4209.90 | 4233.84 | 4236.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 4209.90 | 4233.84 | 4236.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 4180.00 | 4223.07 | 4231.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 4202.80 | 4186.90 | 4201.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 4202.80 | 4186.90 | 4201.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 4208.10 | 4191.14 | 4202.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:15:00 | 4221.50 | 4191.14 | 4202.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 4228.20 | 4198.55 | 4204.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 4236.90 | 4198.55 | 4204.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4224.10 | 4209.97 | 4209.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 4255.00 | 4226.75 | 4218.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 4240.90 | 4241.37 | 4230.62 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 4200.90 | 4227.13 | 4227.54 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 4241.80 | 4229.65 | 4228.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 4256.00 | 4238.57 | 4233.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 4248.50 | 4258.52 | 4248.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 4241.20 | 4258.52 | 4248.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 10:15:00 | 4271.00 | 4261.02 | 4250.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:15:00 | 4275.50 | 4261.02 | 4250.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:00:00 | 4277.90 | 4265.90 | 4255.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 4277.00 | 4315.28 | 4315.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 4277.00 | 4315.28 | 4315.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 4239.00 | 4300.02 | 4308.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 4278.30 | 4274.99 | 4293.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 4278.30 | 4274.99 | 4293.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 4194.20 | 4201.91 | 4232.55 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 4268.20 | 4243.75 | 4241.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 10:15:00 | 4281.50 | 4251.30 | 4244.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 4253.20 | 4260.12 | 4252.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:45:00 | 4253.90 | 4260.12 | 4252.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 4235.00 | 4255.10 | 4250.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 4132.50 | 4255.10 | 4250.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4152.50 | 4234.58 | 4241.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 13:15:00 | 4127.10 | 4178.42 | 4209.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 4196.10 | 4178.38 | 4201.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 11:15:00 | 4172.50 | 4200.44 | 4203.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 11:15:00 | 4101.50 | 4090.87 | 4090.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 4101.50 | 4090.87 | 4090.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 12:15:00 | 4128.20 | 4098.34 | 4093.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4067.00 | 4112.04 | 4103.85 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4077.00 | 4097.87 | 4098.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4062.00 | 4087.39 | 4093.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 4130.50 | 4087.07 | 4090.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:00:00 | 4130.50 | 4087.07 | 4090.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 4138.60 | 4097.37 | 4095.08 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 3946.40 | 4079.37 | 4090.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 3930.10 | 4049.52 | 4075.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3921.30 | 3907.06 | 3963.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 3918.50 | 3907.06 | 3963.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 4013.00 | 3929.61 | 3956.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 4013.00 | 3929.61 | 3956.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 4059.90 | 3955.67 | 3965.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 4059.90 | 3955.67 | 3965.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 4091.00 | 3982.73 | 3977.01 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3961.30 | 3994.95 | 3998.54 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 4083.20 | 4003.49 | 3994.26 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 3954.00 | 4008.45 | 4009.94 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 4071.10 | 4018.14 | 4013.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 4100.40 | 4034.59 | 4021.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 12:15:00 | 4435.80 | 4438.96 | 4361.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:00:00 | 4435.80 | 4438.96 | 4361.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4444.30 | 4473.75 | 4433.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 10:30:00 | 4500.40 | 4473.35 | 4453.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 4512.10 | 4473.35 | 4453.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 12:45:00 | 4499.50 | 4484.09 | 4462.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 13:30:00 | 4498.80 | 4490.37 | 4467.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4450.50 | 4492.04 | 4478.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 4450.50 | 4492.04 | 4478.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 4449.20 | 4483.47 | 4476.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:45:00 | 4449.90 | 4483.47 | 4476.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 4462.00 | 4473.33 | 4472.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:15:00 | 4466.00 | 4473.33 | 4472.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 4425.70 | 4468.01 | 4470.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 4425.70 | 4468.01 | 4470.71 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 14:15:00 | 4528.50 | 4474.51 | 4471.49 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 4461.40 | 4484.53 | 4486.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 09:15:00 | 4436.10 | 4462.34 | 4473.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 4471.90 | 4461.96 | 4471.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:00:00 | 4471.90 | 4461.96 | 4471.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 4446.60 | 4458.89 | 4469.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:15:00 | 4427.20 | 4456.68 | 4465.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:15:00 | 4430.40 | 4454.39 | 4463.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 09:30:00 | 4433.10 | 4410.60 | 4431.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:45:00 | 4436.00 | 4424.90 | 4431.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 4443.30 | 4428.58 | 4432.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:45:00 | 4448.00 | 4428.58 | 4432.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 4436.30 | 4430.12 | 4433.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 4440.00 | 4430.12 | 4433.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 4468.90 | 4437.88 | 4436.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 4468.90 | 4437.88 | 4436.45 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 4418.80 | 4437.01 | 4437.42 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 4461.00 | 4439.09 | 4438.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4467.10 | 4444.69 | 4440.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 4436.80 | 4443.67 | 4441.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 4436.80 | 4443.67 | 4441.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 4436.20 | 4442.18 | 4440.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:30:00 | 4423.90 | 4442.18 | 4440.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 4443.70 | 4442.48 | 4440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 4428.80 | 4442.48 | 4440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 4439.90 | 4441.97 | 4440.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 4389.80 | 4441.97 | 4440.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 4370.10 | 4427.59 | 4434.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 15:15:00 | 4360.00 | 4380.06 | 4396.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 13:15:00 | 4372.50 | 4370.87 | 4384.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 14:00:00 | 4372.50 | 4370.87 | 4384.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4390.00 | 4374.16 | 4383.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4394.10 | 4374.16 | 4383.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4323.70 | 4364.07 | 4378.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:00:00 | 4301.40 | 4351.53 | 4371.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 09:45:00 | 4308.10 | 4340.09 | 4356.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:30:00 | 4310.50 | 4311.55 | 4333.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 4305.20 | 4311.55 | 4333.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4301.00 | 4309.19 | 4328.46 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 4537.20 | 4348.29 | 4339.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 4537.20 | 4348.29 | 4339.53 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:15:00 | 3273.25 | 2024-05-14 10:15:00 | 3300.95 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-05-16 13:30:00 | 3258.10 | 2024-05-16 14:15:00 | 3332.85 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2024-05-16 14:00:00 | 3262.60 | 2024-05-16 14:15:00 | 3332.85 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2024-05-27 11:30:00 | 3426.90 | 2024-05-28 12:15:00 | 3402.40 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-05-28 10:30:00 | 3418.45 | 2024-05-28 12:15:00 | 3402.40 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-06-03 12:00:00 | 3259.00 | 2024-06-04 12:15:00 | 3096.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:30:00 | 3257.55 | 2024-06-04 12:15:00 | 3094.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:00:00 | 3258.20 | 2024-06-04 12:15:00 | 3095.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:45:00 | 3261.95 | 2024-06-04 12:15:00 | 3098.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:00:00 | 3259.00 | 2024-06-04 14:15:00 | 3254.20 | STOP_HIT | 0.50 | 0.15% |
| SELL | retest2 | 2024-06-03 12:30:00 | 3257.55 | 2024-06-04 14:15:00 | 3254.20 | STOP_HIT | 0.50 | 0.10% |
| SELL | retest2 | 2024-06-03 14:00:00 | 3258.20 | 2024-06-04 14:15:00 | 3254.20 | STOP_HIT | 0.50 | 0.12% |
| SELL | retest2 | 2024-06-03 14:45:00 | 3261.95 | 2024-06-04 14:15:00 | 3254.20 | STOP_HIT | 0.50 | 0.24% |
| BUY | retest2 | 2024-06-07 11:30:00 | 3380.00 | 2024-06-12 14:15:00 | 3382.15 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2024-06-27 13:15:00 | 3370.60 | 2024-06-28 09:15:00 | 3395.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-07-15 09:15:00 | 3241.40 | 2024-07-15 09:15:00 | 3220.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-18 14:15:00 | 3262.45 | 2024-07-29 14:15:00 | 3406.10 | STOP_HIT | 1.00 | 4.40% |
| BUY | retest2 | 2024-07-19 11:15:00 | 3265.00 | 2024-07-29 14:15:00 | 3406.10 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2024-07-19 12:00:00 | 3260.00 | 2024-07-29 14:15:00 | 3406.10 | STOP_HIT | 1.00 | 4.48% |
| BUY | retest2 | 2024-07-22 10:30:00 | 3261.10 | 2024-07-29 14:15:00 | 3406.10 | STOP_HIT | 1.00 | 4.45% |
| BUY | retest2 | 2024-07-23 12:00:00 | 3378.90 | 2024-07-29 14:15:00 | 3406.10 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2024-07-31 15:15:00 | 3464.65 | 2024-08-01 12:15:00 | 3450.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-08-01 14:15:00 | 3472.85 | 2024-08-02 09:15:00 | 3418.95 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-08-09 10:45:00 | 3310.80 | 2024-08-13 10:15:00 | 3350.10 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-08-12 09:15:00 | 3310.25 | 2024-08-13 10:15:00 | 3350.10 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-08-26 09:15:00 | 3590.80 | 2024-08-27 14:15:00 | 3550.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-09-03 11:15:00 | 3590.40 | 2024-09-17 09:15:00 | 3729.40 | STOP_HIT | 1.00 | 3.87% |
| BUY | retest2 | 2024-09-04 10:45:00 | 3590.10 | 2024-09-17 09:15:00 | 3729.40 | STOP_HIT | 1.00 | 3.88% |
| BUY | retest2 | 2024-09-04 12:00:00 | 3598.75 | 2024-09-17 09:15:00 | 3729.40 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2024-09-23 09:30:00 | 3815.20 | 2024-09-25 09:15:00 | 3762.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-23 12:30:00 | 3812.95 | 2024-09-25 09:15:00 | 3762.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-09-24 09:45:00 | 3815.30 | 2024-09-25 09:15:00 | 3762.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-09-24 12:00:00 | 3812.00 | 2024-09-25 09:15:00 | 3762.60 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-10-08 09:15:00 | 3546.50 | 2024-10-14 15:15:00 | 3508.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2024-10-08 10:00:00 | 3545.20 | 2024-10-14 15:15:00 | 3508.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2024-10-08 10:30:00 | 3545.00 | 2024-10-14 15:15:00 | 3508.00 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2024-10-09 11:30:00 | 3538.40 | 2024-10-14 15:15:00 | 3508.00 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-10-17 11:30:00 | 3434.30 | 2024-10-25 10:15:00 | 3262.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 11:30:00 | 3434.30 | 2024-10-28 09:15:00 | 3283.05 | STOP_HIT | 0.50 | 4.40% |
| SELL | retest2 | 2024-11-06 09:15:00 | 3129.35 | 2024-11-11 10:15:00 | 3217.50 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2024-11-27 13:15:00 | 3308.25 | 2024-11-28 10:15:00 | 3251.80 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-12-02 09:15:00 | 3236.10 | 2024-12-02 09:15:00 | 3261.85 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-12-10 10:30:00 | 3508.65 | 2024-12-12 09:15:00 | 3434.45 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-12-20 11:15:00 | 3380.95 | 2024-12-23 14:15:00 | 3397.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-01-10 11:15:00 | 3478.00 | 2025-01-10 12:15:00 | 3456.80 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-01-24 11:15:00 | 3414.85 | 2025-01-27 09:15:00 | 3370.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-01-30 15:15:00 | 3371.65 | 2025-02-05 11:15:00 | 3500.30 | STOP_HIT | 1.00 | 3.82% |
| SELL | retest2 | 2025-02-18 10:15:00 | 3202.40 | 2025-02-19 09:15:00 | 3244.35 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-02-19 14:15:00 | 3201.65 | 2025-02-25 11:15:00 | 3208.95 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-02-20 09:15:00 | 3199.55 | 2025-02-25 12:15:00 | 3199.45 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-02-20 10:15:00 | 3199.85 | 2025-02-25 12:15:00 | 3199.45 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-02-25 09:15:00 | 3163.00 | 2025-02-25 12:15:00 | 3199.45 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-03-13 11:30:00 | 3019.10 | 2025-03-18 09:15:00 | 3050.55 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-04-17 10:15:00 | 3270.50 | 2025-05-02 10:15:00 | 3343.90 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-04-17 11:15:00 | 3281.90 | 2025-05-02 10:15:00 | 3343.90 | STOP_HIT | 1.00 | 1.89% |
| SELL | retest2 | 2025-05-05 11:00:00 | 3344.90 | 2025-05-07 13:15:00 | 3348.80 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-05-05 11:30:00 | 3346.10 | 2025-05-07 13:15:00 | 3348.80 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-05-05 12:30:00 | 3346.00 | 2025-05-07 13:15:00 | 3348.80 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-05-05 13:00:00 | 3347.10 | 2025-05-07 13:15:00 | 3348.80 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-09 09:15:00 | 3516.10 | 2025-05-20 13:15:00 | 3597.90 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2025-05-21 12:15:00 | 3577.70 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-05-21 13:30:00 | 3578.30 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-05-21 14:30:00 | 3576.80 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-05-22 09:15:00 | 3537.10 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-05-23 14:30:00 | 3579.40 | 2025-05-26 09:15:00 | 3628.90 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-06-04 09:15:00 | 3498.00 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-06-05 10:30:00 | 3505.00 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-06-05 13:00:00 | 3503.60 | 2025-06-06 12:15:00 | 3537.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-06-11 13:15:00 | 3520.30 | 2025-06-11 14:15:00 | 3539.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3411.20 | 2025-06-18 14:15:00 | 3476.90 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-06-23 12:00:00 | 3489.00 | 2025-07-03 14:15:00 | 3680.90 | STOP_HIT | 1.00 | 5.50% |
| BUY | retest2 | 2025-07-24 10:15:00 | 3481.10 | 2025-07-25 11:15:00 | 3456.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-07-25 09:15:00 | 3477.90 | 2025-07-25 11:15:00 | 3456.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-07-25 13:45:00 | 3476.80 | 2025-07-25 15:15:00 | 3451.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-08-08 09:15:00 | 3442.90 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.32% |
| BUY | retest2 | 2025-08-08 09:45:00 | 3448.30 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.16% |
| BUY | retest2 | 2025-08-11 10:30:00 | 3437.40 | 2025-08-26 14:15:00 | 3591.60 | STOP_HIT | 1.00 | 4.49% |
| BUY | retest2 | 2025-09-08 10:30:00 | 3677.30 | 2025-09-08 12:15:00 | 3659.60 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-09-08 11:15:00 | 3675.80 | 2025-09-08 12:15:00 | 3659.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3540.00 | 2025-09-26 09:15:00 | 3363.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3540.00 | 2025-09-26 10:15:00 | 3393.50 | STOP_HIT | 0.50 | 4.14% |
| BUY | retest2 | 2025-10-06 12:15:00 | 3432.00 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | 2.48% |
| BUY | retest2 | 2025-10-07 10:45:00 | 3430.10 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2025-10-08 09:15:00 | 3551.40 | 2025-10-14 11:15:00 | 3517.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-10-27 12:15:00 | 3746.20 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-27 12:45:00 | 3749.50 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-27 14:15:00 | 3752.30 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-28 09:15:00 | 3760.00 | 2025-10-28 10:15:00 | 3692.10 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-31 09:30:00 | 3777.30 | 2025-11-03 09:15:00 | 3675.30 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2025-11-14 09:30:00 | 3850.00 | 2025-11-14 11:15:00 | 3813.20 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-11-24 09:45:00 | 3925.30 | 2025-11-24 12:15:00 | 3887.40 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-11-25 11:15:00 | 3890.70 | 2025-11-26 10:15:00 | 3910.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-11-26 13:45:00 | 3891.40 | 2025-11-26 14:15:00 | 3902.70 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-11-28 09:15:00 | 3915.90 | 2025-12-01 09:15:00 | 3884.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-11-28 13:15:00 | 3912.10 | 2025-12-01 09:15:00 | 3884.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-16 14:45:00 | 3925.30 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2025-12-18 11:30:00 | 3921.00 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-12-18 12:15:00 | 3918.50 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-12-18 13:00:00 | 3919.20 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2025-12-19 09:15:00 | 3924.40 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-12-24 09:15:00 | 3924.40 | 2025-12-24 09:15:00 | 3916.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-12-30 10:45:00 | 3985.00 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-30 12:00:00 | 3985.10 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2025-12-30 13:00:00 | 3983.70 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 5.01% |
| BUY | retest2 | 2025-12-31 09:15:00 | 4008.90 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 4.35% |
| BUY | retest2 | 2025-12-31 10:15:00 | 4057.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2025-12-31 10:45:00 | 4056.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.12% |
| BUY | retest2 | 2025-12-31 15:00:00 | 4055.00 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2026-01-01 09:15:00 | 4053.50 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2026-01-05 09:30:00 | 4060.60 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2026-01-05 10:15:00 | 4073.70 | 2026-01-12 09:15:00 | 4183.20 | STOP_HIT | 1.00 | 2.69% |
| SELL | retest2 | 2026-01-29 09:15:00 | 3916.20 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-01-30 14:00:00 | 3964.80 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2026-02-01 09:15:00 | 3967.50 | 2026-02-01 12:15:00 | 4044.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-05 13:30:00 | 4091.00 | 2026-02-13 13:15:00 | 4209.90 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-06 11:00:00 | 4087.40 | 2026-02-13 13:15:00 | 4209.90 | STOP_HIT | 1.00 | 3.00% |
| BUY | retest2 | 2026-02-24 11:15:00 | 4275.50 | 2026-03-02 10:15:00 | 4277.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2026-02-24 14:00:00 | 4277.90 | 2026-03-02 10:15:00 | 4277.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2026-03-11 11:15:00 | 4172.50 | 2026-03-18 11:15:00 | 4101.50 | STOP_HIT | 1.00 | 1.70% |
| BUY | retest2 | 2026-04-15 10:30:00 | 4500.40 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-04-15 11:00:00 | 4512.10 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-04-15 12:45:00 | 4499.50 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-04-15 13:30:00 | 4498.80 | 2026-04-17 10:15:00 | 4425.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-04-24 09:15:00 | 4427.20 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-24 10:15:00 | 4430.40 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-04-27 09:30:00 | 4433.10 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-04-27 13:45:00 | 4436.00 | 2026-04-28 09:15:00 | 4468.90 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-05-06 11:00:00 | 4301.40 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.48% |
| SELL | retest2 | 2026-05-07 09:45:00 | 4308.10 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.32% |
| SELL | retest2 | 2026-05-07 14:30:00 | 4310.50 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.26% |
| SELL | retest2 | 2026-05-07 15:00:00 | 4305.20 | 2026-05-08 13:15:00 | 4537.20 | STOP_HIT | 1.00 | -5.39% |
