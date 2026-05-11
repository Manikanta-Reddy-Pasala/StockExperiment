# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 3955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 248 |
| ALERT1 | 152 |
| ALERT2 | 149 |
| ALERT2_SKIP | 80 |
| ALERT3 | 376 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 12 |
| ENTRY2 | 207 |
| PARTIAL | 24 |
| TARGET_HIT | 16 |
| STOP_HIT | 203 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 243 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 97 / 146
- **Target hits / Stop hits / Partials:** 16 / 203 / 24
- **Avg / median % per leg:** 0.83% / -0.56%
- **Sum % (uncompounded):** 201.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 117 | 34 | 29.1% | 13 | 104 | 0 | 0.54% | 63.3% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.25% | -6.3% |
| BUY @ 3rd Alert (retest2) | 112 | 34 | 30.4% | 13 | 99 | 0 | 0.62% | 69.6% |
| SELL (all) | 126 | 63 | 50.0% | 3 | 99 | 24 | 1.10% | 138.0% |
| SELL @ 2nd Alert (retest1) | 11 | 8 | 72.7% | 2 | 5 | 4 | 3.80% | 41.7% |
| SELL @ 3rd Alert (retest2) | 115 | 55 | 47.8% | 1 | 94 | 20 | 0.84% | 96.2% |
| retest1 (combined) | 16 | 8 | 50.0% | 2 | 10 | 4 | 2.22% | 35.5% |
| retest2 (combined) | 227 | 89 | 39.2% | 14 | 193 | 20 | 0.73% | 165.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-25 14:15:00 | 2877.55 | 2902.79 | 2904.22 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 15:15:00 | 2942.10 | 2910.65 | 2907.66 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 09:15:00 | 2845.00 | 2897.52 | 2901.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-26 14:15:00 | 2793.40 | 2849.04 | 2873.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-29 15:15:00 | 2812.40 | 2808.36 | 2835.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-30 09:15:00 | 2813.30 | 2808.36 | 2835.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 2857.00 | 2818.09 | 2837.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-30 10:00:00 | 2857.00 | 2818.09 | 2837.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 2896.00 | 2833.67 | 2842.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 2896.00 | 2833.67 | 2842.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 11:15:00 | 2957.95 | 2858.53 | 2852.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-30 15:15:00 | 2982.00 | 2920.75 | 2887.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-01 13:15:00 | 3040.20 | 3044.47 | 2998.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-01 13:45:00 | 3035.10 | 3044.47 | 2998.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 14:15:00 | 3033.55 | 3042.29 | 3001.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:30:00 | 3044.85 | 3034.49 | 3010.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 12:15:00 | 3049.90 | 3034.49 | 3010.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:30:00 | 3047.00 | 3042.68 | 3024.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 10:15:00 | 3049.00 | 3042.68 | 3024.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 3123.75 | 3135.54 | 3116.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-09 09:15:00 | 3162.00 | 3126.16 | 3119.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-06-15 09:15:00 | 3349.34 | 3300.28 | 3272.13 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 12:15:00 | 3359.10 | 3377.80 | 3379.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 10:15:00 | 3344.15 | 3364.34 | 3371.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 11:15:00 | 3375.00 | 3366.47 | 3371.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 11:15:00 | 3375.00 | 3366.47 | 3371.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 3375.00 | 3366.47 | 3371.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 12:00:00 | 3375.00 | 3366.47 | 3371.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 12:15:00 | 3388.80 | 3370.94 | 3373.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-21 12:45:00 | 3387.95 | 3370.94 | 3373.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 3352.85 | 3367.32 | 3371.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:15:00 | 3350.25 | 3367.32 | 3371.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 09:45:00 | 3348.00 | 3353.54 | 3363.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 12:00:00 | 3343.50 | 3354.17 | 3362.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 09:45:00 | 3331.95 | 3321.51 | 3332.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 3330.45 | 3323.30 | 3332.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:30:00 | 3338.05 | 3323.30 | 3332.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 11:15:00 | 3318.20 | 3322.28 | 3331.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 14:30:00 | 3309.95 | 3322.82 | 3329.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-26 15:15:00 | 3305.00 | 3322.82 | 3329.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 09:45:00 | 3314.95 | 3318.83 | 3326.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 10:15:00 | 3312.25 | 3318.83 | 3326.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 15:15:00 | 3318.95 | 3302.50 | 3312.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-28 09:15:00 | 3281.10 | 3302.50 | 3312.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 3323.85 | 3306.77 | 3313.45 | SL hit (close>static) qty=1.00 sl=3319.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-07-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 10:15:00 | 3224.65 | 3183.41 | 3180.24 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 3194.15 | 3199.68 | 3199.75 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 3228.30 | 3202.90 | 3200.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 11:15:00 | 3249.55 | 3215.45 | 3207.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-14 15:15:00 | 3350.00 | 3365.53 | 3342.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 15:15:00 | 3350.00 | 3365.53 | 3342.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 3350.00 | 3365.53 | 3342.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 3370.45 | 3365.53 | 3342.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 3357.75 | 3363.97 | 3344.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 3393.30 | 3364.31 | 3352.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 10:00:00 | 3393.90 | 3370.23 | 3356.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-25 12:15:00 | 3498.00 | 3518.26 | 3518.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 12:15:00 | 3498.00 | 3518.26 | 3518.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-25 13:15:00 | 3483.35 | 3511.28 | 3515.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-26 09:15:00 | 3557.05 | 3515.28 | 3515.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-26 09:15:00 | 3557.05 | 3515.28 | 3515.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 09:15:00 | 3557.05 | 3515.28 | 3515.65 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-26 10:15:00 | 3558.45 | 3523.91 | 3519.54 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 13:15:00 | 3488.95 | 3512.11 | 3515.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 15:15:00 | 3470.00 | 3498.72 | 3508.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 3519.90 | 3502.96 | 3509.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 3519.90 | 3502.96 | 3509.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 3519.90 | 3502.96 | 3509.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 11:30:00 | 3489.95 | 3499.13 | 3506.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 12:15:00 | 3483.30 | 3499.13 | 3506.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 14:30:00 | 3478.75 | 3495.02 | 3502.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 09:15:00 | 3478.75 | 3495.61 | 3501.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 09:15:00 | 3516.10 | 3499.71 | 3503.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 09:45:00 | 3504.60 | 3499.71 | 3503.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-28 10:15:00 | 3523.35 | 3504.44 | 3505.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-28 11:00:00 | 3523.35 | 3504.44 | 3505.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-28 11:15:00 | 3536.05 | 3510.76 | 3507.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 11:15:00 | 3536.05 | 3510.76 | 3507.87 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 13:15:00 | 3450.55 | 3501.21 | 3506.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 12:15:00 | 3402.20 | 3475.65 | 3491.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-02 09:15:00 | 3434.50 | 3432.58 | 3462.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-02 09:30:00 | 3442.95 | 3432.58 | 3462.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 3460.65 | 3438.19 | 3462.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:00:00 | 3460.65 | 3438.19 | 3462.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 11:15:00 | 3479.20 | 3446.40 | 3463.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 11:45:00 | 3460.00 | 3446.40 | 3463.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 12:15:00 | 3485.00 | 3454.12 | 3465.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 12:45:00 | 3481.90 | 3454.12 | 3465.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 13:15:00 | 3444.95 | 3452.28 | 3463.87 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 3540.85 | 3471.48 | 3469.98 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 3425.00 | 3465.47 | 3468.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-04 15:15:00 | 3405.00 | 3424.34 | 3438.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 09:15:00 | 3466.00 | 3432.67 | 3441.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-07 09:15:00 | 3466.00 | 3432.67 | 3441.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 3466.00 | 3432.67 | 3441.35 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 12:15:00 | 3468.00 | 3449.89 | 3447.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-07 14:15:00 | 3536.95 | 3470.17 | 3457.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 09:15:00 | 3576.95 | 3585.91 | 3556.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-10 10:00:00 | 3576.95 | 3585.91 | 3556.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 10:15:00 | 3572.75 | 3583.28 | 3557.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 10:30:00 | 3560.00 | 3583.28 | 3557.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 3564.20 | 3578.92 | 3562.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:00:00 | 3564.20 | 3578.92 | 3562.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 3588.95 | 3580.92 | 3564.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:30:00 | 3565.45 | 3580.92 | 3564.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 15:15:00 | 3565.00 | 3577.74 | 3564.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-11 09:15:00 | 3616.15 | 3577.74 | 3564.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 09:30:00 | 3599.00 | 3600.86 | 3587.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-14 10:30:00 | 3606.00 | 3602.03 | 3589.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 10:15:00 | 3595.75 | 3606.43 | 3605.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-17 10:15:00 | 3584.65 | 3602.07 | 3603.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 10:15:00 | 3584.65 | 3602.07 | 3603.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 11:15:00 | 3567.60 | 3595.18 | 3600.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 11:15:00 | 3612.00 | 3564.33 | 3577.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 11:15:00 | 3612.00 | 3564.33 | 3577.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 3612.00 | 3564.33 | 3577.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 12:00:00 | 3612.00 | 3564.33 | 3577.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 12:15:00 | 3608.10 | 3573.09 | 3579.89 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 3620.15 | 3588.40 | 3586.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 15:15:00 | 3659.00 | 3602.52 | 3592.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-21 14:15:00 | 3612.40 | 3619.80 | 3607.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 14:15:00 | 3612.40 | 3619.80 | 3607.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 3612.40 | 3619.80 | 3607.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 14:45:00 | 3594.60 | 3619.80 | 3607.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 3599.70 | 3615.78 | 3606.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:45:00 | 3646.75 | 3612.37 | 3608.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 14:30:00 | 3643.95 | 3630.80 | 3620.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 14:45:00 | 3640.05 | 3634.61 | 3627.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 09:15:00 | 3660.00 | 3634.89 | 3628.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 3658.40 | 3639.59 | 3631.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-28 09:15:00 | 3707.00 | 3646.25 | 3638.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 3721.20 | 3692.47 | 3685.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 3681.15 | 3693.12 | 3694.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 3681.15 | 3693.12 | 3694.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 12:15:00 | 3638.85 | 3682.27 | 3689.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 11:15:00 | 3664.25 | 3646.59 | 3663.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 11:15:00 | 3664.25 | 3646.59 | 3663.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 11:15:00 | 3664.25 | 3646.59 | 3663.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 11:45:00 | 3665.70 | 3646.59 | 3663.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 3675.00 | 3652.27 | 3664.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 13:00:00 | 3675.00 | 3652.27 | 3664.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 13:15:00 | 3698.95 | 3661.61 | 3668.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 14:00:00 | 3698.95 | 3661.61 | 3668.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 15:15:00 | 3693.35 | 3669.67 | 3670.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-06 09:15:00 | 3666.00 | 3669.67 | 3670.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-08 10:15:00 | 3665.85 | 3649.72 | 3649.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 10:15:00 | 3665.85 | 3649.72 | 3649.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-08 11:15:00 | 3675.00 | 3654.77 | 3651.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 09:15:00 | 3715.85 | 3748.89 | 3717.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-12 09:15:00 | 3715.85 | 3748.89 | 3717.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 3715.85 | 3748.89 | 3717.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 3699.90 | 3748.89 | 3717.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 3711.20 | 3741.35 | 3717.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 3708.00 | 3741.35 | 3717.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 11:15:00 | 3699.00 | 3732.88 | 3715.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:00:00 | 3699.00 | 3732.88 | 3715.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 12:15:00 | 3690.00 | 3724.31 | 3713.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 12:45:00 | 3688.40 | 3724.31 | 3713.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 15:15:00 | 3678.20 | 3704.55 | 3706.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 3611.10 | 3685.86 | 3697.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 14:15:00 | 3649.25 | 3643.31 | 3668.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-13 15:00:00 | 3649.25 | 3643.31 | 3668.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 3630.65 | 3638.65 | 3661.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-14 12:15:00 | 3610.00 | 3631.95 | 3654.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-15 14:15:00 | 3686.15 | 3594.97 | 3612.76 | SL hit (close>static) qty=1.00 sl=3683.95 alert=retest2 |

### Cycle 22 — BUY (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 12:15:00 | 3637.40 | 3623.62 | 3622.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-18 13:15:00 | 3659.45 | 3630.79 | 3625.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 15:15:00 | 3610.50 | 3627.71 | 3625.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 15:15:00 | 3610.50 | 3627.71 | 3625.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 15:15:00 | 3610.50 | 3627.71 | 3625.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-20 09:15:00 | 3610.20 | 3627.71 | 3625.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 3565.20 | 3615.21 | 3619.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 3552.00 | 3602.56 | 3613.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 13:15:00 | 3559.35 | 3551.39 | 3572.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-21 14:00:00 | 3559.35 | 3551.39 | 3572.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 3561.75 | 3553.46 | 3571.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 3561.75 | 3553.46 | 3571.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 3506.10 | 3543.44 | 3563.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 10:15:00 | 3492.70 | 3543.44 | 3563.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 13:00:00 | 3500.05 | 3519.60 | 3546.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 09:15:00 | 3503.80 | 3440.66 | 3438.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-28 09:15:00 | 3503.80 | 3440.66 | 3438.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 10:15:00 | 3533.65 | 3500.36 | 3477.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 3515.55 | 3519.99 | 3495.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 15:00:00 | 3515.55 | 3519.99 | 3495.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 3587.00 | 3532.60 | 3505.46 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 15:15:00 | 3498.35 | 3523.12 | 3523.80 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-10-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 09:15:00 | 3554.75 | 3529.44 | 3526.61 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 10:15:00 | 3505.10 | 3524.57 | 3524.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 12:15:00 | 3478.95 | 3512.33 | 3518.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 15:15:00 | 3447.00 | 3446.15 | 3470.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-09 09:15:00 | 3415.00 | 3446.15 | 3470.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 13:15:00 | 3444.30 | 3420.65 | 3433.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-10 13:15:00 | 3444.30 | 3420.65 | 3433.23 | SL hit (close>ema400) qty=1.00 sl=3433.23 alert=retest1 |

### Cycle 28 — BUY (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 13:15:00 | 3430.25 | 3402.09 | 3399.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 3471.00 | 3415.88 | 3405.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 12:15:00 | 3501.75 | 3505.34 | 3472.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 12:45:00 | 3499.40 | 3505.34 | 3472.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 3543.90 | 3536.64 | 3515.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-20 10:30:00 | 3575.00 | 3543.32 | 3520.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 12:15:00 | 3471.15 | 3525.21 | 3515.78 | SL hit (close<static) qty=1.00 sl=3513.40 alert=retest2 |

### Cycle 29 — SELL (started 2023-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 14:15:00 | 3466.80 | 3501.74 | 3506.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 3427.15 | 3479.05 | 3493.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 15:15:00 | 3476.00 | 3430.55 | 3446.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-25 15:15:00 | 3476.00 | 3430.55 | 3446.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-25 15:15:00 | 3476.00 | 3430.55 | 3446.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 3366.55 | 3430.55 | 3446.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 10:15:00 | 3486.40 | 3442.85 | 3440.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2023-10-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 10:15:00 | 3486.40 | 3442.85 | 3440.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 11:15:00 | 3502.40 | 3454.76 | 3445.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 11:15:00 | 3492.50 | 3498.44 | 3477.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 11:30:00 | 3495.30 | 3498.44 | 3477.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 12:15:00 | 3462.60 | 3491.27 | 3476.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-30 13:00:00 | 3462.60 | 3491.27 | 3476.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 3473.65 | 3487.75 | 3475.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 15:00:00 | 3487.45 | 3487.69 | 3476.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-31 09:15:00 | 3484.60 | 3486.15 | 3477.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-07 09:15:00 | 3578.85 | 3678.63 | 3690.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2023-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-07 09:15:00 | 3578.85 | 3678.63 | 3690.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 10:15:00 | 3549.85 | 3652.87 | 3677.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 14:15:00 | 3614.00 | 3578.24 | 3605.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 14:15:00 | 3614.00 | 3578.24 | 3605.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 14:15:00 | 3614.00 | 3578.24 | 3605.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 15:00:00 | 3614.00 | 3578.24 | 3605.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 15:15:00 | 3581.30 | 3578.86 | 3603.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:15:00 | 3578.40 | 3578.86 | 3603.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 3574.05 | 3577.89 | 3600.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-09 10:15:00 | 3562.00 | 3577.89 | 3600.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-10 10:15:00 | 3629.50 | 3606.57 | 3605.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 10:15:00 | 3629.50 | 3606.57 | 3605.14 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 14:15:00 | 3558.05 | 3603.02 | 3604.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 15:15:00 | 3555.00 | 3593.42 | 3600.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 13:15:00 | 3588.15 | 3581.15 | 3589.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 13:15:00 | 3588.15 | 3581.15 | 3589.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 13:15:00 | 3588.15 | 3581.15 | 3589.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 14:00:00 | 3588.15 | 3581.15 | 3589.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 14:15:00 | 3602.30 | 3585.38 | 3590.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 15:00:00 | 3602.30 | 3585.38 | 3590.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 3598.65 | 3588.03 | 3591.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 3652.35 | 3588.03 | 3591.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2023-11-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 09:15:00 | 3625.05 | 3595.44 | 3594.40 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 09:15:00 | 3563.95 | 3590.75 | 3594.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-17 11:15:00 | 3535.85 | 3565.94 | 3578.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 3479.40 | 3450.32 | 3468.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 3479.40 | 3450.32 | 3468.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 3479.40 | 3450.32 | 3468.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 09:45:00 | 3474.00 | 3450.32 | 3468.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 10:15:00 | 3490.00 | 3458.26 | 3470.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-23 10:45:00 | 3496.15 | 3458.26 | 3470.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2023-11-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 13:15:00 | 3507.90 | 3477.66 | 3477.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 10:15:00 | 3512.00 | 3492.34 | 3484.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 11:15:00 | 3532.10 | 3537.68 | 3517.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-28 12:00:00 | 3532.10 | 3537.68 | 3517.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 12:15:00 | 3578.00 | 3545.74 | 3522.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 10:15:00 | 3580.00 | 3552.62 | 3532.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-29 10:45:00 | 3581.90 | 3561.10 | 3538.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 12:15:00 | 3694.00 | 3716.80 | 3716.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 12:15:00 | 3694.00 | 3716.80 | 3716.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 10:15:00 | 3668.00 | 3699.59 | 3707.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 3667.05 | 3631.55 | 3655.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 3667.05 | 3631.55 | 3655.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 3667.05 | 3631.55 | 3655.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 09:15:00 | 3587.25 | 3619.96 | 3634.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-18 10:00:00 | 3577.10 | 3611.39 | 3629.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 14:15:00 | 3574.90 | 3546.69 | 3544.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-12-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 14:15:00 | 3574.90 | 3546.69 | 3544.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 3579.50 | 3558.63 | 3551.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 3650.00 | 3652.49 | 3624.68 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:30:00 | 3689.25 | 3660.17 | 3630.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 3678.60 | 3677.80 | 3656.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 12:15:00 | 3692.75 | 3678.12 | 3658.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:00:00 | 3692.30 | 3684.11 | 3665.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 15:15:00 | 3690.00 | 3684.33 | 3666.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-02 09:15:00 | 3612.50 | 3670.87 | 3663.83 | SL hit (close<ema400) qty=1.00 sl=3663.83 alert=retest1 |

### Cycle 39 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 3642.75 | 3658.10 | 3658.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-03 09:15:00 | 3601.75 | 3637.82 | 3648.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-04 10:15:00 | 3610.00 | 3594.28 | 3615.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-04 11:00:00 | 3610.00 | 3594.28 | 3615.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 11:15:00 | 3605.05 | 3596.43 | 3614.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-04 12:30:00 | 3588.95 | 3592.61 | 3611.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-05 10:30:00 | 3588.90 | 3596.14 | 3606.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-05 11:15:00 | 3621.70 | 3601.25 | 3607.51 | SL hit (close>static) qty=1.00 sl=3617.45 alert=retest2 |

### Cycle 40 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 3617.70 | 3609.85 | 3608.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 10:15:00 | 3634.45 | 3614.77 | 3611.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 13:15:00 | 3617.80 | 3621.12 | 3615.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-09 14:00:00 | 3617.80 | 3621.12 | 3615.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 14:15:00 | 3626.00 | 3622.09 | 3616.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-09 14:30:00 | 3614.40 | 3622.09 | 3616.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 15:15:00 | 3610.20 | 3619.72 | 3615.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:15:00 | 3648.65 | 3619.72 | 3615.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 3750.00 | 3645.77 | 3628.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 10:15:00 | 3823.00 | 3645.77 | 3628.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 11:15:00 | 3768.45 | 3668.62 | 3640.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 11:00:00 | 3774.25 | 3733.86 | 3693.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-12 11:45:00 | 3768.60 | 3742.20 | 3719.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-12 15:15:00 | 3716.00 | 3740.60 | 3726.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 09:30:00 | 3755.70 | 3743.05 | 3728.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 11:15:00 | 3755.20 | 3744.56 | 3730.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-15 13:15:00 | 3755.00 | 3747.65 | 3734.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-15 14:15:00 | 3704.65 | 3739.52 | 3733.27 | SL hit (close<static) qty=1.00 sl=3714.80 alert=retest2 |

### Cycle 41 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 3709.00 | 3725.72 | 3727.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 3670.25 | 3714.63 | 3722.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-16 14:15:00 | 3714.95 | 3714.06 | 3720.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-16 15:00:00 | 3714.95 | 3714.06 | 3720.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 3728.95 | 3717.04 | 3721.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 3678.80 | 3717.04 | 3721.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 13:15:00 | 3730.55 | 3711.03 | 3715.07 | SL hit (close>static) qty=1.00 sl=3728.95 alert=retest2 |

### Cycle 42 — BUY (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 15:15:00 | 3744.00 | 3717.51 | 3717.31 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 3626.95 | 3699.40 | 3709.09 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 3716.50 | 3692.40 | 3692.19 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 15:15:00 | 3666.30 | 3693.41 | 3695.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 3656.30 | 3681.34 | 3689.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 09:15:00 | 3656.90 | 3656.76 | 3671.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 3656.90 | 3656.76 | 3671.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 3656.90 | 3656.76 | 3671.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 10:00:00 | 3656.90 | 3656.76 | 3671.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 3673.95 | 3658.82 | 3670.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:00:00 | 3673.95 | 3658.82 | 3670.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 12:15:00 | 3682.35 | 3663.52 | 3671.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 12:45:00 | 3686.80 | 3663.52 | 3671.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 13:15:00 | 3719.70 | 3674.76 | 3675.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 14:00:00 | 3719.70 | 3674.76 | 3675.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 14:15:00 | 3738.05 | 3687.42 | 3681.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 15:15:00 | 3900.00 | 3729.93 | 3701.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 09:15:00 | 3993.55 | 3999.59 | 3923.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 09:45:00 | 4000.05 | 3999.59 | 3923.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 13:15:00 | 3945.85 | 3986.18 | 3941.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 13:45:00 | 3945.65 | 3986.18 | 3941.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 3900.00 | 3968.94 | 3937.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 14:30:00 | 3883.95 | 3968.94 | 3937.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 3867.55 | 3948.66 | 3931.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-31 09:15:00 | 3994.75 | 3948.66 | 3931.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-01 13:15:00 | 4394.23 | 4183.95 | 4091.82 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 13:15:00 | 4195.50 | 4357.35 | 4357.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 14:15:00 | 4130.00 | 4311.88 | 4336.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 10:15:00 | 3934.90 | 3929.32 | 4005.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-12 11:00:00 | 3934.90 | 3929.32 | 4005.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 3969.15 | 3925.68 | 3951.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 14:45:00 | 3983.60 | 3925.68 | 3951.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 3977.00 | 3935.95 | 3954.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-14 09:15:00 | 3942.35 | 3935.95 | 3954.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-19 13:15:00 | 3952.15 | 3894.21 | 3890.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 13:15:00 | 3952.15 | 3894.21 | 3890.89 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 3879.90 | 3892.10 | 3892.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 15:15:00 | 3865.00 | 3880.57 | 3886.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 15:15:00 | 3759.00 | 3744.93 | 3771.58 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-26 09:15:00 | 3726.20 | 3744.93 | 3771.58 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 10:15:00 | 3759.95 | 3728.17 | 3743.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-27 10:15:00 | 3759.95 | 3728.17 | 3743.67 | SL hit (close>ema400) qty=1.00 sl=3743.67 alert=retest1 |

### Cycle 50 — BUY (started 2024-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 14:15:00 | 3799.45 | 3756.85 | 3753.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 15:15:00 | 3807.00 | 3766.88 | 3758.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 09:15:00 | 3764.55 | 3766.41 | 3759.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-28 09:45:00 | 3763.00 | 3766.41 | 3759.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 3727.00 | 3758.53 | 3756.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 3727.00 | 3758.53 | 3756.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 3722.75 | 3751.37 | 3753.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 3694.50 | 3740.00 | 3747.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-01 09:15:00 | 3700.00 | 3672.68 | 3695.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 09:15:00 | 3700.00 | 3672.68 | 3695.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 3700.00 | 3672.68 | 3695.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:45:00 | 3705.55 | 3672.68 | 3695.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 10:15:00 | 3705.80 | 3679.30 | 3696.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 10:30:00 | 3725.00 | 3679.30 | 3696.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 12:15:00 | 3700.35 | 3687.45 | 3697.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 13:15:00 | 3709.30 | 3687.45 | 3697.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 3683.25 | 3689.65 | 3696.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:15:00 | 3707.00 | 3689.65 | 3696.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 09:15:00 | 3713.25 | 3694.37 | 3697.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 09:45:00 | 3713.25 | 3694.37 | 3697.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 3713.00 | 3698.09 | 3699.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-02 11:30:00 | 3711.05 | 3698.09 | 3699.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-02 12:15:00 | 3720.00 | 3702.48 | 3701.20 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 09:15:00 | 3677.10 | 3697.40 | 3699.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 10:15:00 | 3668.90 | 3691.70 | 3696.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 3715.80 | 3692.54 | 3695.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 13:15:00 | 3715.80 | 3692.54 | 3695.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 3715.80 | 3692.54 | 3695.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-04 13:45:00 | 3708.05 | 3692.54 | 3695.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 3711.15 | 3696.26 | 3696.63 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2024-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 15:15:00 | 3715.00 | 3700.01 | 3698.30 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 3683.45 | 3696.70 | 3696.95 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 10:15:00 | 3720.00 | 3701.36 | 3699.05 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 11:15:00 | 3653.50 | 3694.90 | 3698.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 15:15:00 | 3640.00 | 3669.52 | 3683.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 13:15:00 | 3682.85 | 3660.45 | 3672.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 13:15:00 | 3682.85 | 3660.45 | 3672.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 3682.85 | 3660.45 | 3672.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 14:00:00 | 3682.85 | 3660.45 | 3672.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 3700.00 | 3668.36 | 3675.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 15:00:00 | 3700.00 | 3668.36 | 3675.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 3710.00 | 3676.69 | 3678.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 3672.75 | 3676.69 | 3678.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 13:00:00 | 3666.00 | 3671.46 | 3675.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 15:15:00 | 3709.95 | 3675.87 | 3673.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 15:15:00 | 3709.95 | 3675.87 | 3673.45 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 3638.10 | 3665.77 | 3669.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 3580.00 | 3648.61 | 3661.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 14:15:00 | 3635.05 | 3627.04 | 3646.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-13 15:00:00 | 3635.05 | 3627.04 | 3646.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 3587.55 | 3614.14 | 3628.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 3575.50 | 3614.14 | 3628.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 09:15:00 | 3582.80 | 3618.87 | 3625.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 13:15:00 | 3637.35 | 3618.24 | 3617.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-19 13:15:00 | 3637.35 | 3618.24 | 3617.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 14:15:00 | 3649.60 | 3624.51 | 3620.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-20 09:15:00 | 3617.75 | 3627.08 | 3622.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 09:15:00 | 3617.75 | 3627.08 | 3622.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 3617.75 | 3627.08 | 3622.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-20 09:45:00 | 3617.20 | 3627.08 | 3622.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 3643.90 | 3630.44 | 3624.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 12:30:00 | 3656.55 | 3633.98 | 3627.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 15:00:00 | 3649.70 | 3638.67 | 3630.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:15:00 | 3668.15 | 3640.44 | 3632.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-02 09:15:00 | 4022.21 | 3954.35 | 3902.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 13:15:00 | 3930.30 | 3967.99 | 3969.58 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 14:15:00 | 3983.00 | 3971.00 | 3970.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 09:15:00 | 3996.60 | 3977.55 | 3973.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 14:15:00 | 3997.50 | 3999.62 | 3988.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 14:15:00 | 3997.50 | 3999.62 | 3988.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 3997.50 | 3999.62 | 3988.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 14:45:00 | 3985.95 | 3999.62 | 3988.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 15:15:00 | 3971.00 | 3993.89 | 3986.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 4075.35 | 3993.89 | 3986.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-10 14:15:00 | 3999.00 | 4052.26 | 4057.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 14:15:00 | 3999.00 | 4052.26 | 4057.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 09:15:00 | 3983.80 | 4031.01 | 4046.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 3835.70 | 3834.44 | 3874.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-18 09:15:00 | 3835.70 | 3834.44 | 3874.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 3835.70 | 3834.44 | 3874.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:15:00 | 3827.00 | 3835.00 | 3861.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-19 14:45:00 | 3825.20 | 3836.66 | 3847.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 09:30:00 | 3827.05 | 3841.07 | 3844.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 10:30:00 | 3824.55 | 3839.48 | 3843.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 3851.30 | 3841.84 | 3844.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:45:00 | 3853.15 | 3841.84 | 3844.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 3845.00 | 3842.47 | 3844.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-04-24 09:15:00 | 3919.90 | 3857.12 | 3850.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 09:15:00 | 3919.90 | 3857.12 | 3850.54 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 3796.50 | 3880.07 | 3882.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 14:15:00 | 3775.05 | 3801.47 | 3818.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 10:15:00 | 3755.45 | 3745.22 | 3769.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-06 11:00:00 | 3755.45 | 3745.22 | 3769.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 3773.70 | 3750.92 | 3770.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:45:00 | 3774.25 | 3750.92 | 3770.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 12:15:00 | 3770.35 | 3754.80 | 3770.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 13:15:00 | 3764.50 | 3754.80 | 3770.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-06 14:30:00 | 3761.95 | 3761.43 | 3770.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-07 09:15:00 | 3852.15 | 3777.75 | 3776.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 09:15:00 | 3852.15 | 3777.75 | 3776.55 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 3785.00 | 3790.77 | 3790.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 09:15:00 | 3771.05 | 3786.56 | 3788.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 3800.00 | 3778.32 | 3782.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 15:15:00 | 3800.00 | 3778.32 | 3782.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 3800.00 | 3778.32 | 3782.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 09:15:00 | 3799.75 | 3778.32 | 3782.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 3762.85 | 3775.22 | 3780.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 10:30:00 | 3723.30 | 3765.39 | 3775.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 11:15:00 | 3792.65 | 3778.56 | 3777.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 3792.65 | 3778.56 | 3777.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 15:15:00 | 3820.00 | 3791.76 | 3785.09 | Break + close above crossover candle high |

### Cycle 69 — SELL (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 09:15:00 | 3720.00 | 3777.41 | 3779.17 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 10:15:00 | 3793.85 | 3777.26 | 3775.51 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 3760.00 | 3774.20 | 3774.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 3743.85 | 3768.13 | 3771.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 15:15:00 | 3764.00 | 3762.80 | 3768.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:15:00 | 3757.00 | 3762.80 | 3768.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 3749.55 | 3760.15 | 3766.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:30:00 | 3759.35 | 3760.15 | 3766.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 3792.95 | 3761.71 | 3765.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 3792.95 | 3761.71 | 3765.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 3766.25 | 3762.62 | 3765.39 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 3790.10 | 3771.02 | 3768.87 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 11:15:00 | 3747.30 | 3770.81 | 3770.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 3693.95 | 3741.00 | 3755.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 09:15:00 | 3670.25 | 3668.38 | 3702.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 3670.25 | 3668.38 | 3702.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 3670.25 | 3668.38 | 3702.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:30:00 | 3700.60 | 3668.38 | 3702.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 3704.75 | 3677.39 | 3700.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 3704.75 | 3677.39 | 3700.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 3690.00 | 3679.92 | 3699.48 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 13:15:00 | 3734.80 | 3704.35 | 3702.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 14:15:00 | 3735.35 | 3710.55 | 3705.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 3686.75 | 3709.43 | 3705.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 3686.75 | 3709.43 | 3705.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 3686.75 | 3709.43 | 3705.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 3683.00 | 3709.43 | 3705.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 3699.20 | 3707.38 | 3705.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 11:45:00 | 3719.80 | 3709.82 | 3706.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 13:15:00 | 3717.15 | 3708.94 | 3706.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:45:00 | 3708.50 | 3708.73 | 3707.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 13:15:00 | 3707.45 | 3707.49 | 3707.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 13:15:00 | 3700.00 | 3705.99 | 3706.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 3700.00 | 3705.99 | 3706.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 3681.15 | 3701.02 | 3704.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 13:15:00 | 3667.90 | 3656.77 | 3670.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 13:15:00 | 3667.90 | 3656.77 | 3670.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 3667.90 | 3656.77 | 3670.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:30:00 | 3665.20 | 3656.77 | 3670.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 14:15:00 | 3694.40 | 3664.30 | 3673.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 15:00:00 | 3694.40 | 3664.30 | 3673.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 3663.05 | 3664.05 | 3672.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:45:00 | 3724.50 | 3676.14 | 3676.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 3724.45 | 3685.80 | 3681.20 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 10:15:00 | 3671.00 | 3678.88 | 3679.81 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 3692.50 | 3682.50 | 3681.35 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 3648.50 | 3677.29 | 3679.52 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 3693.95 | 3680.87 | 3679.62 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 3650.00 | 3675.36 | 3677.37 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 3724.80 | 3670.75 | 3670.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 09:15:00 | 3756.65 | 3721.22 | 3701.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 3749.80 | 3760.38 | 3736.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 3746.90 | 3760.38 | 3736.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 3770.00 | 3795.88 | 3771.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:45:00 | 3771.00 | 3795.88 | 3771.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 3761.50 | 3789.00 | 3770.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 3762.20 | 3789.00 | 3770.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 3808.05 | 3792.81 | 3773.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 12:30:00 | 3843.00 | 3801.72 | 3779.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 15:15:00 | 3834.00 | 3811.10 | 3787.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 10:15:00 | 3832.05 | 3817.95 | 3795.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 11:00:00 | 3832.95 | 3820.95 | 3798.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-19 10:15:00 | 4227.30 | 4119.61 | 4023.74 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 4212.00 | 4229.45 | 4230.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 4170.20 | 4217.60 | 4225.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 11:15:00 | 4213.60 | 4213.02 | 4221.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 11:15:00 | 4213.60 | 4213.02 | 4221.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 4213.60 | 4213.02 | 4221.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 4216.00 | 4213.02 | 4221.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 4227.05 | 4215.83 | 4222.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:30:00 | 4232.75 | 4215.83 | 4222.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 4234.40 | 4219.54 | 4223.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:30:00 | 4242.65 | 4219.54 | 4223.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 4218.70 | 4219.38 | 4222.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:30:00 | 4214.90 | 4219.38 | 4222.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 4207.40 | 4216.28 | 4220.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 4170.15 | 4210.28 | 4217.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 13:15:00 | 4184.10 | 4161.29 | 4177.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 10:15:00 | 4237.10 | 4191.02 | 4186.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 4237.10 | 4191.02 | 4186.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 12:15:00 | 4247.70 | 4225.73 | 4215.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 4231.05 | 4236.91 | 4224.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 4231.05 | 4236.91 | 4224.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4231.05 | 4236.91 | 4224.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 15:15:00 | 4274.00 | 4250.71 | 4242.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 09:15:00 | 4204.65 | 4245.22 | 4241.31 | SL hit (close<static) qty=1.00 sl=4215.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 4197.75 | 4235.73 | 4237.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 4188.80 | 4226.34 | 4232.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 14:15:00 | 4213.60 | 4210.93 | 4223.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 4213.60 | 4210.93 | 4223.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 4239.60 | 4218.27 | 4224.43 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 4322.55 | 4242.45 | 4233.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 14:15:00 | 4405.00 | 4274.96 | 4249.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 13:15:00 | 4293.50 | 4309.27 | 4283.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 13:15:00 | 4293.50 | 4309.27 | 4283.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 4293.50 | 4309.27 | 4283.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 14:00:00 | 4293.50 | 4309.27 | 4283.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 4292.50 | 4305.92 | 4284.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 4292.50 | 4305.92 | 4284.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 4299.00 | 4304.54 | 4285.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 4284.00 | 4304.54 | 4285.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 4257.90 | 4295.21 | 4282.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:15:00 | 4245.55 | 4295.21 | 4282.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 4246.20 | 4285.41 | 4279.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 4244.55 | 4285.41 | 4279.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2024-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 12:15:00 | 4260.25 | 4274.61 | 4275.32 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 14:15:00 | 4298.05 | 4276.17 | 4275.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4322.00 | 4288.03 | 4281.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 11:15:00 | 4280.00 | 4291.53 | 4284.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 11:15:00 | 4280.00 | 4291.53 | 4284.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 4280.00 | 4291.53 | 4284.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 4280.00 | 4291.53 | 4284.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 4252.60 | 4283.75 | 4281.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:00:00 | 4252.60 | 4283.75 | 4281.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 13:15:00 | 4258.30 | 4278.66 | 4279.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 09:15:00 | 4233.75 | 4266.39 | 4273.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 4251.90 | 4230.14 | 4246.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 4251.90 | 4230.14 | 4246.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 4251.90 | 4230.14 | 4246.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 4205.15 | 4224.96 | 4235.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 4309.95 | 4242.65 | 4239.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 09:15:00 | 4309.95 | 4242.65 | 4239.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-19 10:15:00 | 4594.85 | 4313.09 | 4272.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 12:15:00 | 4309.50 | 4313.63 | 4279.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 12:15:00 | 4309.50 | 4313.63 | 4279.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 4309.50 | 4313.63 | 4279.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 09:30:00 | 4371.45 | 4331.89 | 4297.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 12:00:00 | 4393.20 | 4352.60 | 4313.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:15:00 | 4367.95 | 4353.97 | 4317.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 14:15:00 | 4309.05 | 4321.65 | 4321.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 4309.05 | 4321.65 | 4321.81 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 4341.40 | 4324.21 | 4322.86 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 4281.15 | 4326.28 | 4326.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 11:15:00 | 4275.30 | 4308.20 | 4317.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 4289.15 | 4278.73 | 4297.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 4289.15 | 4278.73 | 4297.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 4289.15 | 4278.73 | 4297.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 4290.25 | 4278.73 | 4297.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 4300.00 | 4282.98 | 4297.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 4300.00 | 4282.98 | 4297.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 4313.85 | 4289.16 | 4299.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:45:00 | 4320.00 | 4289.16 | 4299.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 4349.80 | 4307.34 | 4305.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 14:15:00 | 4375.65 | 4321.00 | 4312.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 4631.00 | 4659.37 | 4594.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 15:00:00 | 4631.00 | 4659.37 | 4594.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 4611.00 | 4642.77 | 4602.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:00:00 | 4611.00 | 4642.77 | 4602.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 4540.00 | 4622.22 | 4596.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 4540.00 | 4622.22 | 4596.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 4554.80 | 4608.74 | 4592.93 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 4517.45 | 4577.53 | 4580.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 4417.80 | 4497.75 | 4530.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4435.00 | 4406.10 | 4454.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4435.00 | 4406.10 | 4454.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4435.00 | 4406.10 | 4454.68 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 4548.00 | 4465.99 | 4464.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 10:15:00 | 4639.70 | 4500.73 | 4480.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 11:15:00 | 4494.75 | 4499.54 | 4481.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 11:15:00 | 4494.75 | 4499.54 | 4481.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 4494.75 | 4499.54 | 4481.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 4494.75 | 4499.54 | 4481.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 4465.95 | 4492.82 | 4480.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:30:00 | 4424.95 | 4492.82 | 4480.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 4470.00 | 4488.26 | 4479.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 4584.00 | 4485.27 | 4479.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 11:15:00 | 4551.70 | 4684.71 | 4687.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 4551.70 | 4684.71 | 4687.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 09:15:00 | 4437.55 | 4471.42 | 4508.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 4424.80 | 4412.19 | 4443.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 4424.80 | 4412.19 | 4443.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 4424.80 | 4412.19 | 4443.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:30:00 | 4431.30 | 4412.19 | 4443.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 4390.00 | 4409.61 | 4437.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 4444.65 | 4409.61 | 4437.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4431.75 | 4414.04 | 4436.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:15:00 | 4428.50 | 4414.04 | 4436.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 11:45:00 | 4420.75 | 4417.79 | 4434.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:00:00 | 4426.50 | 4434.29 | 4436.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 4387.00 | 4398.37 | 4415.69 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 10:15:00 | 4396.50 | 4398.00 | 4413.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 11:00:00 | 4396.50 | 4398.00 | 4413.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 4342.15 | 4340.87 | 4362.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 13:00:00 | 4323.10 | 4338.27 | 4356.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 4328.95 | 4322.66 | 4340.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 09:15:00 | 4392.55 | 4332.19 | 4335.58 | SL hit (close>static) qty=1.00 sl=4390.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 4396.05 | 4344.96 | 4341.07 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 4324.90 | 4369.93 | 4372.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 4256.55 | 4341.59 | 4358.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 10:15:00 | 4313.40 | 4284.86 | 4310.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 10:15:00 | 4313.40 | 4284.86 | 4310.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 4313.40 | 4284.86 | 4310.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 11:00:00 | 4313.40 | 4284.86 | 4310.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 4334.05 | 4294.70 | 4313.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:00:00 | 4334.05 | 4294.70 | 4313.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 4332.10 | 4302.18 | 4314.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 13:00:00 | 4332.10 | 4302.18 | 4314.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 4391.80 | 4327.39 | 4324.52 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 4285.40 | 4327.63 | 4330.81 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 4342.00 | 4329.26 | 4328.32 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 10:15:00 | 4319.35 | 4329.59 | 4329.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 10:15:00 | 4289.50 | 4308.07 | 4317.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 4309.90 | 4308.43 | 4316.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 11:15:00 | 4309.90 | 4308.43 | 4316.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 4309.90 | 4308.43 | 4316.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:30:00 | 4312.85 | 4308.43 | 4316.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 12:15:00 | 4317.30 | 4310.21 | 4316.66 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 15:15:00 | 4335.00 | 4321.53 | 4320.85 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 4306.75 | 4319.15 | 4319.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 4278.00 | 4308.01 | 4314.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 11:15:00 | 4294.55 | 4263.41 | 4277.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 11:15:00 | 4294.55 | 4263.41 | 4277.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 4294.55 | 4263.41 | 4277.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:45:00 | 4295.25 | 4263.41 | 4277.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 12:15:00 | 4294.90 | 4269.71 | 4279.21 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 4316.50 | 4290.07 | 4286.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 10:15:00 | 4363.45 | 4312.97 | 4300.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 4364.95 | 4375.85 | 4343.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 10:00:00 | 4364.95 | 4375.85 | 4343.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 4301.00 | 4357.60 | 4340.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 4301.00 | 4357.60 | 4340.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 4324.20 | 4350.92 | 4339.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 4335.45 | 4342.27 | 4336.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 4311.25 | 4332.50 | 4333.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 4311.25 | 4332.50 | 4333.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 4286.00 | 4319.18 | 4325.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 4319.90 | 4319.33 | 4325.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 4319.90 | 4319.33 | 4325.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 4319.90 | 4319.33 | 4325.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:45:00 | 4324.15 | 4319.33 | 4325.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 4376.85 | 4330.83 | 4329.96 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 4283.00 | 4324.29 | 4327.54 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 13:15:00 | 4334.00 | 4328.17 | 4327.81 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 4320.00 | 4327.45 | 4327.60 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 09:15:00 | 4378.05 | 4337.57 | 4332.19 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 13:15:00 | 4310.80 | 4326.37 | 4328.09 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 4348.85 | 4330.87 | 4329.97 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 15:15:00 | 4319.00 | 4328.49 | 4328.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 4305.05 | 4322.71 | 4326.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 14:15:00 | 4265.05 | 4252.02 | 4277.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 15:00:00 | 4265.05 | 4252.02 | 4277.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 4195.70 | 4241.23 | 4268.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 10:30:00 | 4160.00 | 4222.28 | 4257.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 13:00:00 | 4170.35 | 4157.95 | 4173.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 11:15:00 | 4146.70 | 4103.63 | 4098.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 11:15:00 | 4146.70 | 4103.63 | 4098.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 4162.10 | 4135.03 | 4117.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 4173.60 | 4184.96 | 4156.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 4173.60 | 4184.96 | 4156.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 4173.60 | 4184.96 | 4156.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 4173.60 | 4184.96 | 4156.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 4146.55 | 4177.27 | 4155.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 4146.55 | 4177.27 | 4155.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 4218.05 | 4185.43 | 4161.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:30:00 | 4151.60 | 4185.43 | 4161.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 4182.05 | 4210.22 | 4185.85 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2024-10-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 15:15:00 | 4155.00 | 4175.02 | 4176.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 4114.90 | 4163.00 | 4170.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 3939.00 | 3922.00 | 3968.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 09:15:00 | 3886.05 | 3922.00 | 3968.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 3925.90 | 3896.47 | 3927.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 3868.80 | 3896.47 | 3927.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 11:45:00 | 3873.20 | 3883.76 | 3913.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:00:00 | 3877.85 | 3883.31 | 3905.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 09:30:00 | 3860.00 | 3875.60 | 3898.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 3892.40 | 3875.09 | 3888.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 3892.40 | 3875.09 | 3888.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 3918.95 | 3883.86 | 3891.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 09:30:00 | 3874.60 | 3885.64 | 3891.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 11:15:00 | 3929.70 | 3896.41 | 3895.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 11:15:00 | 3929.70 | 3896.41 | 3895.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 15:15:00 | 3949.00 | 3909.33 | 3902.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 3903.00 | 3908.06 | 3902.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 3903.00 | 3908.06 | 3902.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 3903.00 | 3908.06 | 3902.56 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 3874.30 | 3899.41 | 3899.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 13:15:00 | 3841.50 | 3887.82 | 3894.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 15:15:00 | 3881.00 | 3874.97 | 3886.82 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-04 09:15:00 | 3796.60 | 3869.18 | 3881.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:15:00 | 3799.05 | 3814.12 | 3839.36 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:15:00 | 3606.77 | 3684.40 | 3725.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:15:00 | 3609.10 | 3684.40 | 3725.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-11-14 15:15:00 | 3419.15 | 3481.13 | 3520.57 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 120 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 3492.75 | 3433.00 | 3429.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 3529.05 | 3452.21 | 3438.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 3478.90 | 3479.59 | 3458.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 3478.90 | 3479.59 | 3458.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 3470.80 | 3476.02 | 3466.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 14:00:00 | 3470.80 | 3476.02 | 3466.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 3489.05 | 3478.62 | 3468.37 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 12:15:00 | 3450.00 | 3462.09 | 3463.40 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 3470.50 | 3462.68 | 3462.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 14:15:00 | 3482.40 | 3468.92 | 3465.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 09:15:00 | 3444.75 | 3465.00 | 3464.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 3444.75 | 3465.00 | 3464.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 3444.75 | 3465.00 | 3464.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 3444.75 | 3465.00 | 3464.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 10:15:00 | 3455.00 | 3463.00 | 3463.45 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 3481.95 | 3465.92 | 3464.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 14:15:00 | 3500.05 | 3474.83 | 3469.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 3462.40 | 3476.27 | 3470.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 3462.40 | 3476.27 | 3470.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 3462.40 | 3476.27 | 3470.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 3465.70 | 3476.27 | 3470.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 3476.75 | 3476.37 | 3471.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 3495.80 | 3472.98 | 3471.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 3463.40 | 3491.16 | 3494.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 3463.40 | 3491.16 | 3494.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 15:15:00 | 3458.80 | 3472.10 | 3481.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 09:15:00 | 3440.55 | 3439.41 | 3455.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 3440.55 | 3439.41 | 3455.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 3440.55 | 3439.41 | 3455.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:45:00 | 3455.15 | 3439.41 | 3455.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 3470.00 | 3447.53 | 3456.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 11:45:00 | 3469.95 | 3447.53 | 3456.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 12:15:00 | 3465.00 | 3451.02 | 3457.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:30:00 | 3455.85 | 3451.12 | 3457.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 15:00:00 | 3455.00 | 3451.89 | 3456.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 3512.40 | 3466.61 | 3462.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 3512.40 | 3466.61 | 3462.89 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 3464.55 | 3491.02 | 3492.83 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 3506.20 | 3496.09 | 3494.77 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 14:15:00 | 3482.60 | 3493.40 | 3493.67 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 3507.00 | 3496.12 | 3494.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 09:15:00 | 3513.15 | 3499.52 | 3496.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 3477.90 | 3495.20 | 3494.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 3477.90 | 3495.20 | 3494.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 3477.90 | 3495.20 | 3494.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 3477.90 | 3495.20 | 3494.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 3492.50 | 3494.66 | 3494.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 15:00:00 | 3505.30 | 3498.52 | 3496.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 12:15:00 | 3490.90 | 3496.08 | 3496.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 3490.90 | 3496.08 | 3496.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 3463.75 | 3487.36 | 3492.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 3341.85 | 3331.59 | 3364.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-24 10:00:00 | 3341.85 | 3331.59 | 3364.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 11:15:00 | 3358.95 | 3339.69 | 3362.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 11:45:00 | 3359.35 | 3339.69 | 3362.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 3360.55 | 3343.86 | 3362.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 13:00:00 | 3360.55 | 3343.86 | 3362.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 3395.20 | 3354.13 | 3365.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:00:00 | 3395.20 | 3354.13 | 3365.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 14:15:00 | 3406.80 | 3364.66 | 3369.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 14:45:00 | 3405.05 | 3364.66 | 3369.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 3356.55 | 3366.22 | 3369.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:15:00 | 3343.10 | 3363.51 | 3367.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 3401.85 | 3369.70 | 3368.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 3401.85 | 3369.70 | 3368.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 11:15:00 | 3420.00 | 3384.66 | 3375.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 3395.05 | 3403.84 | 3390.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 3395.05 | 3403.84 | 3390.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 3395.05 | 3403.84 | 3390.56 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 10:15:00 | 3375.60 | 3390.20 | 3391.20 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 3406.00 | 3392.27 | 3391.31 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 09:15:00 | 3377.35 | 3389.28 | 3390.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-02 10:15:00 | 3363.00 | 3384.03 | 3387.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-02 11:15:00 | 3384.80 | 3384.18 | 3387.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-02 11:15:00 | 3384.80 | 3384.18 | 3387.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 11:15:00 | 3384.80 | 3384.18 | 3387.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 12:00:00 | 3384.80 | 3384.18 | 3387.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 12:15:00 | 3404.55 | 3388.25 | 3388.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-02 13:00:00 | 3404.55 | 3388.25 | 3388.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 13:15:00 | 3397.60 | 3390.12 | 3389.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 3411.35 | 3396.66 | 3392.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 3393.70 | 3398.69 | 3394.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 11:15:00 | 3393.70 | 3398.69 | 3394.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 3393.70 | 3398.69 | 3394.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:00:00 | 3393.70 | 3398.69 | 3394.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 3396.90 | 3398.33 | 3394.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:15:00 | 3396.45 | 3398.33 | 3394.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 3385.00 | 3395.66 | 3393.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:45:00 | 3388.50 | 3395.66 | 3393.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 3383.15 | 3393.16 | 3393.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 3383.15 | 3393.16 | 3393.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 3375.00 | 3389.53 | 3391.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 09:15:00 | 3351.15 | 3381.85 | 3387.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 14:15:00 | 3348.50 | 3348.08 | 3358.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 15:00:00 | 3348.50 | 3348.08 | 3358.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 3390.00 | 3354.84 | 3359.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:00:00 | 3390.00 | 3354.84 | 3359.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 3383.20 | 3360.52 | 3361.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:45:00 | 3395.65 | 3360.52 | 3361.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 11:15:00 | 3377.00 | 3363.81 | 3363.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 3403.90 | 3375.65 | 3368.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 13:15:00 | 3409.75 | 3409.79 | 3393.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 13:45:00 | 3403.20 | 3409.79 | 3393.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 3431.40 | 3414.11 | 3396.77 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 3361.15 | 3391.16 | 3392.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 3322.00 | 3370.60 | 3381.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 11:15:00 | 3329.00 | 3320.63 | 3341.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:45:00 | 3323.80 | 3320.63 | 3341.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 3378.05 | 3332.12 | 3344.83 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 3368.75 | 3352.66 | 3352.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 10:15:00 | 3397.80 | 3366.58 | 3358.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 15:15:00 | 3490.00 | 3490.52 | 3461.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:45:00 | 3530.80 | 3496.07 | 3466.61 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:15:00 | 3523.50 | 3496.07 | 3466.61 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 12:00:00 | 3530.05 | 3504.51 | 3475.66 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3489.50 | 3524.48 | 3502.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 3489.50 | 3524.48 | 3502.84 | SL hit (close<ema400) qty=1.00 sl=3502.84 alert=retest1 |

### Cycle 141 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 3625.00 | 3634.65 | 3635.02 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 3649.05 | 3634.75 | 3634.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 3688.20 | 3654.10 | 3644.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 3657.60 | 3669.96 | 3658.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 3657.60 | 3669.96 | 3658.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3657.60 | 3669.96 | 3658.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 3657.60 | 3669.96 | 3658.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 3662.20 | 3668.41 | 3658.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 3693.00 | 3666.73 | 3658.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 14:15:00 | 3650.80 | 3657.41 | 3657.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 14:15:00 | 3650.80 | 3657.41 | 3657.98 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 3689.55 | 3661.75 | 3658.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 13:15:00 | 3730.10 | 3683.57 | 3670.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 3676.05 | 3700.43 | 3682.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 3676.05 | 3700.43 | 3682.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 3676.05 | 3700.43 | 3682.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 3676.05 | 3700.43 | 3682.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 3680.15 | 3696.38 | 3682.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:15:00 | 3678.70 | 3696.38 | 3682.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 3672.85 | 3691.67 | 3681.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 3652.55 | 3691.67 | 3681.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 3686.35 | 3690.61 | 3682.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 14:30:00 | 3706.95 | 3688.18 | 3682.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 15:15:00 | 3650.00 | 3680.54 | 3679.61 | SL hit (close<static) qty=1.00 sl=3671.60 alert=retest2 |

### Cycle 145 — SELL (started 2025-02-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 09:15:00 | 3658.95 | 3676.22 | 3677.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 3633.40 | 3650.35 | 3658.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 15:15:00 | 3573.95 | 3548.04 | 3580.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 10:30:00 | 3523.50 | 3542.82 | 3571.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-13 12:15:00 | 3527.90 | 3541.36 | 3568.67 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 3520.00 | 3527.71 | 3550.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 3497.40 | 3512.28 | 3541.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 3347.32 | 3436.16 | 3485.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 3351.51 | 3436.16 | 3485.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-02-17 12:15:00 | 3445.50 | 3430.86 | 3469.72 | SL hit (close>ema200) qty=0.50 sl=3430.86 alert=retest1 |

### Cycle 146 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 3192.55 | 3159.83 | 3155.41 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 3131.55 | 3160.46 | 3162.17 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 3227.00 | 3160.73 | 3151.82 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 3126.55 | 3148.83 | 3149.04 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 3208.65 | 3159.07 | 3152.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 3220.00 | 3177.97 | 3163.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 10:15:00 | 3200.00 | 3201.29 | 3186.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 11:00:00 | 3200.00 | 3201.29 | 3186.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 3242.30 | 3240.26 | 3224.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:30:00 | 3227.70 | 3240.26 | 3224.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 3357.60 | 3361.82 | 3339.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 3363.30 | 3361.82 | 3339.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 3365.00 | 3365.33 | 3350.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 3327.60 | 3365.33 | 3350.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 3310.30 | 3354.33 | 3347.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:30:00 | 3315.20 | 3354.33 | 3347.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 3346.15 | 3352.69 | 3347.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 11:15:00 | 3352.00 | 3352.69 | 3347.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 14:15:00 | 3362.70 | 3352.10 | 3348.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 3404.70 | 3351.11 | 3348.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 13:15:00 | 3333.20 | 3346.80 | 3347.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 3333.20 | 3346.80 | 3347.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 3308.00 | 3332.87 | 3340.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 12:15:00 | 3312.10 | 3301.26 | 3316.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 12:15:00 | 3312.10 | 3301.26 | 3316.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 3312.10 | 3301.26 | 3316.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 3312.10 | 3301.26 | 3316.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 3316.30 | 3304.27 | 3316.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:30:00 | 3317.40 | 3304.27 | 3316.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 3306.90 | 3304.79 | 3315.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 3306.90 | 3304.79 | 3315.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 3292.90 | 3299.25 | 3310.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 09:45:00 | 3268.00 | 3297.62 | 3305.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 3104.60 | 3189.93 | 3241.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 13:15:00 | 3112.65 | 3100.33 | 3141.79 | SL hit (close>ema200) qty=0.50 sl=3100.33 alert=retest2 |

### Cycle 152 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 3161.25 | 3130.21 | 3128.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 3179.90 | 3144.36 | 3135.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 13:15:00 | 3162.50 | 3168.98 | 3154.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 13:45:00 | 3165.00 | 3168.98 | 3154.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 14:15:00 | 3172.90 | 3169.77 | 3155.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 14:45:00 | 3159.30 | 3169.77 | 3155.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 15:15:00 | 3165.00 | 3168.81 | 3156.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 09:45:00 | 3186.60 | 3172.03 | 3159.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:45:00 | 3187.20 | 3185.48 | 3171.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-17 09:15:00 | 3151.10 | 3176.92 | 3170.09 | SL hit (close<static) qty=1.00 sl=3155.70 alert=retest2 |

### Cycle 153 — SELL (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 14:15:00 | 3144.70 | 3163.41 | 3165.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 09:15:00 | 3116.90 | 3151.02 | 3159.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 3137.30 | 3136.25 | 3145.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-22 09:15:00 | 3137.30 | 3136.25 | 3145.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 09:15:00 | 3137.30 | 3136.25 | 3145.81 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 12:15:00 | 3171.30 | 3155.65 | 3153.52 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 09:15:00 | 3135.80 | 3152.17 | 3152.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-23 10:15:00 | 3129.20 | 3147.58 | 3150.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 09:15:00 | 3179.40 | 3142.02 | 3144.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 09:15:00 | 3179.40 | 3142.02 | 3144.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 3179.40 | 3142.02 | 3144.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 3179.40 | 3142.02 | 3144.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 3160.20 | 3145.66 | 3145.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 3237.70 | 3178.07 | 3164.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 13:15:00 | 3182.50 | 3189.62 | 3175.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-28 13:45:00 | 3182.20 | 3189.62 | 3175.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 3171.30 | 3185.96 | 3174.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:30:00 | 3155.60 | 3185.96 | 3174.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 3169.00 | 3182.57 | 3174.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 3162.30 | 3185.19 | 3176.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 3208.40 | 3203.56 | 3191.51 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 3155.00 | 3189.58 | 3189.92 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 3203.70 | 3192.40 | 3191.18 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 11:15:00 | 3147.50 | 3183.57 | 3187.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 3120.00 | 3170.85 | 3181.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 3163.50 | 3138.89 | 3157.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 3163.50 | 3138.89 | 3157.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 3163.50 | 3138.89 | 3157.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:45:00 | 3162.50 | 3138.89 | 3157.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 3175.30 | 3146.17 | 3159.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:00:00 | 3175.30 | 3146.17 | 3159.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 3171.90 | 3151.32 | 3160.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 3171.90 | 3151.32 | 3160.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 3169.90 | 3161.40 | 3163.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 3200.70 | 3161.40 | 3163.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 160 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 3191.50 | 3167.42 | 3166.05 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 14:15:00 | 3121.20 | 3160.58 | 3164.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 15:15:00 | 3110.00 | 3150.47 | 3159.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 3146.40 | 3132.98 | 3142.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 3146.40 | 3132.98 | 3142.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 3146.40 | 3132.98 | 3142.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 3152.60 | 3132.98 | 3142.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 3145.10 | 3135.40 | 3142.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:45:00 | 3128.50 | 3135.33 | 3141.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 3180.30 | 3134.56 | 3133.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3180.30 | 3134.56 | 3133.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 3215.20 | 3150.69 | 3140.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 3227.00 | 3227.48 | 3196.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 12:30:00 | 3220.00 | 3227.48 | 3196.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 3256.80 | 3256.68 | 3236.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 3303.60 | 3267.38 | 3243.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:30:00 | 3303.10 | 3279.46 | 3253.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 3310.50 | 3313.30 | 3294.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 15:15:00 | 3287.50 | 3303.17 | 3304.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 3287.50 | 3303.17 | 3304.14 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3318.40 | 3306.21 | 3305.44 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 12:15:00 | 3294.30 | 3304.95 | 3305.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 3282.20 | 3298.79 | 3302.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 13:15:00 | 3245.00 | 3244.43 | 3264.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:00:00 | 3245.00 | 3244.43 | 3264.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 3317.70 | 3259.09 | 3269.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:30:00 | 3309.80 | 3259.09 | 3269.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 3314.90 | 3270.25 | 3273.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 3331.70 | 3270.25 | 3273.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 3397.60 | 3295.72 | 3284.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 3450.10 | 3397.90 | 3375.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 3502.20 | 3506.09 | 3459.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:00:00 | 3502.20 | 3506.09 | 3459.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3482.10 | 3496.60 | 3466.12 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 3429.70 | 3462.19 | 3464.87 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 3511.00 | 3468.34 | 3465.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 3541.90 | 3495.94 | 3480.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 3502.30 | 3506.91 | 3494.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 3498.60 | 3506.91 | 3494.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 3500.00 | 3505.46 | 3497.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 3499.60 | 3505.46 | 3497.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 3497.20 | 3503.81 | 3497.18 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 3493.50 | 3503.04 | 3503.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 3453.30 | 3492.45 | 3498.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 3469.00 | 3457.59 | 3471.23 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-16 09:15:00 | 3406.40 | 3457.59 | 3471.23 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 3441.70 | 3418.68 | 3426.22 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 3441.70 | 3418.68 | 3426.22 | SL hit (close>ema400) qty=1.00 sl=3426.22 alert=retest1 |

### Cycle 170 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 3341.70 | 3312.95 | 3310.03 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 15:15:00 | 3303.90 | 3321.92 | 3323.93 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 3328.30 | 3325.67 | 3325.36 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 3297.30 | 3319.99 | 3322.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 3290.00 | 3307.52 | 3315.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3325.00 | 3278.73 | 3290.53 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 13:15:00 | 3319.60 | 3299.51 | 3298.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 12:15:00 | 3332.70 | 3317.26 | 3309.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 14:15:00 | 3455.90 | 3462.53 | 3426.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 3455.90 | 3462.53 | 3426.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 3429.10 | 3454.48 | 3428.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 3430.70 | 3454.48 | 3428.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 3423.80 | 3448.35 | 3428.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 3421.20 | 3448.35 | 3428.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 3415.70 | 3441.82 | 3427.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 3413.70 | 3441.82 | 3427.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 3419.40 | 3437.33 | 3426.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 3412.20 | 3437.33 | 3426.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 3425.90 | 3435.05 | 3426.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 3438.70 | 3434.16 | 3426.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 3433.90 | 3433.33 | 3427.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 09:45:00 | 3442.00 | 3431.46 | 3426.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 3390.70 | 3423.31 | 3423.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 3390.70 | 3423.31 | 3423.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 11:15:00 | 3374.50 | 3413.55 | 3419.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 3425.00 | 3370.75 | 3383.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 3430.10 | 3370.75 | 3383.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3401.00 | 3376.80 | 3384.68 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 13:15:00 | 3393.10 | 3389.60 | 3389.42 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 09:15:00 | 3383.00 | 3388.81 | 3389.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 10:15:00 | 3372.90 | 3385.63 | 3387.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 3384.80 | 3384.24 | 3386.44 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 09:15:00 | 3413.80 | 3388.29 | 3387.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 12:15:00 | 3434.40 | 3404.00 | 3395.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 3413.60 | 3422.75 | 3408.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 3413.60 | 3422.75 | 3408.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 3406.10 | 3419.42 | 3408.56 | EMA400 retest candle locked (from upside) |

### Cycle 179 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 3371.30 | 3401.21 | 3402.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 3369.30 | 3394.83 | 3399.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 3352.70 | 3343.30 | 3354.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 3357.80 | 3343.30 | 3354.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 3358.30 | 3346.30 | 3354.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:45:00 | 3354.10 | 3346.30 | 3354.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 3353.80 | 3347.80 | 3354.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 3336.10 | 3350.04 | 3353.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 3169.29 | 3221.73 | 3256.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 11:15:00 | 3200.10 | 3191.54 | 3217.93 | SL hit (close>ema200) qty=0.50 sl=3191.54 alert=retest2 |

### Cycle 180 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 3150.00 | 3118.18 | 3114.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 11:15:00 | 3182.40 | 3140.35 | 3126.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3161.80 | 3168.44 | 3148.28 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 3103.10 | 3137.24 | 3140.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 12:15:00 | 3083.10 | 3120.55 | 3132.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 11:15:00 | 3107.00 | 3096.04 | 3112.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 11:45:00 | 3105.60 | 3096.04 | 3112.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 3114.00 | 3099.64 | 3112.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:30:00 | 3110.70 | 3099.64 | 3112.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 3133.20 | 3106.35 | 3114.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:15:00 | 3144.20 | 3106.35 | 3114.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 3110.90 | 3107.26 | 3114.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 3134.60 | 3107.26 | 3114.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 3120.30 | 3109.87 | 3114.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 3131.20 | 3109.87 | 3114.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3107.50 | 3109.39 | 3113.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 3093.10 | 3109.39 | 3113.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 10:15:00 | 3124.60 | 3110.41 | 3110.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 3124.60 | 3110.41 | 3110.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 3131.10 | 3117.88 | 3114.02 | Break + close above crossover candle high |

### Cycle 183 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 3059.70 | 3111.49 | 3112.49 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 15:15:00 | 3120.00 | 3110.85 | 3110.10 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 3100.30 | 3108.74 | 3109.21 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 12:15:00 | 3114.60 | 3110.00 | 3109.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 14:15:00 | 3120.00 | 3112.87 | 3111.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 3105.90 | 3112.93 | 3111.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 3092.40 | 3112.93 | 3111.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 3090.00 | 3108.35 | 3109.55 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 14:15:00 | 3124.60 | 3112.59 | 3110.96 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 3092.40 | 3109.84 | 3111.24 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 3119.90 | 3111.92 | 3111.80 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 3099.00 | 3111.15 | 3112.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 15:15:00 | 3081.00 | 3104.78 | 3108.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 3118.10 | 3107.44 | 3109.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 13:15:00 | 3091.60 | 3110.60 | 3110.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 3094.20 | 3109.48 | 3110.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 15:00:00 | 3091.20 | 3105.83 | 3108.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 3089.40 | 3098.95 | 3104.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 3035.90 | 3031.77 | 3048.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:30:00 | 3053.90 | 3031.77 | 3048.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 3046.40 | 3035.36 | 3047.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:30:00 | 3047.20 | 3035.36 | 3047.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 3051.40 | 3038.57 | 3047.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 3051.40 | 3038.57 | 3047.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 3053.00 | 3041.45 | 3048.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 3069.90 | 3041.45 | 3048.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 3085.70 | 3054.69 | 3053.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 3085.70 | 3054.69 | 3053.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 11:15:00 | 3092.90 | 3062.34 | 3057.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 3102.10 | 3104.98 | 3086.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 3102.10 | 3104.98 | 3086.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 3093.90 | 3104.99 | 3091.63 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 13:15:00 | 3070.00 | 3084.58 | 3084.94 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 3091.10 | 3085.52 | 3085.17 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 3070.00 | 3083.49 | 3084.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 3055.50 | 3076.54 | 3081.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3047.50 | 3037.48 | 3050.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 3047.50 | 3037.48 | 3050.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 3042.00 | 3039.14 | 3047.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:30:00 | 3040.50 | 3039.14 | 3047.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3065.70 | 3042.17 | 3046.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 3061.10 | 3042.17 | 3046.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3064.10 | 3046.56 | 3047.94 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 11:15:00 | 3059.90 | 3049.23 | 3049.03 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 12:15:00 | 3039.00 | 3050.53 | 3050.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 15:15:00 | 3029.10 | 3042.73 | 3046.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3035.60 | 3035.43 | 3042.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 3037.00 | 3035.43 | 3042.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 3051.90 | 3038.72 | 3043.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 3051.90 | 3038.72 | 3043.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 3057.90 | 3042.56 | 3044.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:30:00 | 3055.70 | 3042.56 | 3044.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 13:15:00 | 3077.10 | 3049.47 | 3047.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 14:15:00 | 3090.00 | 3057.57 | 3051.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 12:15:00 | 3100.10 | 3101.52 | 3088.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 13:00:00 | 3100.10 | 3101.52 | 3088.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3110.00 | 3117.67 | 3106.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 3131.00 | 3122.56 | 3109.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 3128.50 | 3131.35 | 3123.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 3069.40 | 3110.52 | 3115.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 14:15:00 | 3069.40 | 3110.52 | 3115.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 3048.60 | 3077.99 | 3094.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3086.30 | 3061.30 | 3072.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 3086.30 | 3061.30 | 3072.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3074.50 | 3063.94 | 3072.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:15:00 | 3059.20 | 3063.94 | 3072.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 10:15:00 | 3086.00 | 3075.58 | 3074.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2025-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 10:15:00 | 3086.00 | 3075.58 | 3074.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 14:15:00 | 3098.50 | 3083.89 | 3079.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 3071.00 | 3082.07 | 3079.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:30:00 | 3077.70 | 3082.07 | 3079.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 3073.10 | 3080.27 | 3078.71 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 3065.10 | 3077.24 | 3077.47 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 09:15:00 | 3112.40 | 3081.74 | 3079.15 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 3061.80 | 3074.67 | 3076.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 14:15:00 | 3052.60 | 3066.35 | 3070.46 | Break + close below crossover candle low |

### Cycle 204 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 3117.00 | 3076.27 | 3074.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 3148.70 | 3090.75 | 3081.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3190.20 | 3201.92 | 3170.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 09:45:00 | 3204.00 | 3201.92 | 3170.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3173.00 | 3196.14 | 3170.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:45:00 | 3173.60 | 3196.14 | 3170.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 3170.00 | 3190.91 | 3170.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 3171.30 | 3190.91 | 3170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 3163.70 | 3185.47 | 3170.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 3163.70 | 3185.47 | 3170.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 3165.00 | 3181.38 | 3169.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:45:00 | 3157.60 | 3181.38 | 3169.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3158.00 | 3174.69 | 3169.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 11:30:00 | 3194.30 | 3178.23 | 3172.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 3195.10 | 3182.58 | 3174.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 3189.00 | 3180.72 | 3176.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:30:00 | 3198.00 | 3187.05 | 3179.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3256.00 | 3264.00 | 3247.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 13:15:00 | 3215.50 | 3239.23 | 3239.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 13:15:00 | 3215.50 | 3239.23 | 3239.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 3201.10 | 3231.59 | 3236.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 3228.30 | 3223.29 | 3229.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:00:00 | 3228.30 | 3223.29 | 3229.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 3232.00 | 3225.03 | 3230.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 3229.40 | 3225.03 | 3230.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 3225.00 | 3225.03 | 3229.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 3227.60 | 3225.03 | 3229.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3225.40 | 3225.10 | 3229.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 3214.70 | 3227.10 | 3229.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 15:15:00 | 3270.00 | 3230.10 | 3229.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 3270.00 | 3230.10 | 3229.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 3333.40 | 3250.76 | 3238.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 3296.00 | 3302.70 | 3278.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-20 09:30:00 | 3323.10 | 3308.41 | 3293.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 3292.90 | 3305.30 | 3293.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 3292.90 | 3305.30 | 3293.29 | SL hit (close<ema400) qty=1.00 sl=3293.29 alert=retest1 |

### Cycle 207 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3330.00 | 3349.16 | 3350.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 3317.90 | 3338.76 | 3345.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3355.50 | 3341.84 | 3345.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:15:00 | 3350.00 | 3341.84 | 3345.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3335.00 | 3340.47 | 3344.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 3328.90 | 3339.18 | 3343.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3316.40 | 3338.35 | 3342.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 3315.00 | 3290.03 | 3288.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 3315.00 | 3290.03 | 3288.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 3327.20 | 3297.47 | 3292.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 3308.10 | 3311.57 | 3301.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-06 10:00:00 | 3308.10 | 3311.57 | 3301.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 3301.40 | 3309.53 | 3301.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 3314.90 | 3305.70 | 3301.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 3285.00 | 3304.05 | 3301.62 | SL hit (close<static) qty=1.00 sl=3295.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 3228.50 | 3288.94 | 3294.98 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 3380.90 | 3287.48 | 3287.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 3431.50 | 3328.13 | 3306.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 3672.20 | 3678.30 | 3627.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 3672.20 | 3678.30 | 3627.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3679.60 | 3726.23 | 3693.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3679.60 | 3726.23 | 3693.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3703.60 | 3721.70 | 3694.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 12:15:00 | 3731.70 | 3717.96 | 3695.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 3716.00 | 3712.04 | 3701.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 3724.60 | 3709.54 | 3703.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 10:00:00 | 3725.00 | 3720.70 | 3710.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3730.20 | 3770.11 | 3744.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 3743.20 | 3770.11 | 3744.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 3719.40 | 3759.97 | 3742.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 3719.40 | 3759.97 | 3742.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 12:15:00 | 3705.50 | 3740.25 | 3735.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:30:00 | 3710.00 | 3740.25 | 3735.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 3703.90 | 3727.36 | 3730.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 3703.90 | 3727.36 | 3730.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 3685.00 | 3718.88 | 3726.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 3722.20 | 3717.92 | 3724.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 3722.20 | 3717.92 | 3724.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 11:15:00 | 3717.00 | 3717.73 | 3723.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 3700.00 | 3719.46 | 3722.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 3701.00 | 3719.46 | 3722.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 3732.20 | 3719.05 | 3721.94 | SL hit (close>static) qty=1.00 sl=3723.70 alert=retest2 |

### Cycle 212 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 3749.40 | 3725.12 | 3724.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 11:15:00 | 3769.90 | 3734.08 | 3728.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 10:15:00 | 3795.00 | 3797.35 | 3769.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:30:00 | 3773.50 | 3797.35 | 3769.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 3772.30 | 3792.47 | 3776.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 3772.30 | 3792.47 | 3776.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 3740.50 | 3782.08 | 3772.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 3779.50 | 3782.08 | 3772.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 3810.50 | 3854.97 | 3857.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 3810.50 | 3854.97 | 3857.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 3803.80 | 3844.74 | 3853.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3738.00 | 3708.76 | 3744.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 3738.00 | 3708.76 | 3744.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 3749.50 | 3716.91 | 3744.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 3749.50 | 3716.91 | 3744.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 3768.50 | 3727.23 | 3747.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:45:00 | 3769.60 | 3727.23 | 3747.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 3793.30 | 3740.44 | 3751.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 3793.30 | 3740.44 | 3751.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 3785.80 | 3761.44 | 3759.68 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 10:15:00 | 3742.60 | 3757.67 | 3758.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 11:15:00 | 3731.20 | 3752.38 | 3755.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 3744.30 | 3674.10 | 3695.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 3744.30 | 3674.10 | 3695.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3734.20 | 3686.12 | 3698.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 3733.20 | 3686.12 | 3698.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 216 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 3734.00 | 3709.53 | 3707.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 3748.80 | 3717.38 | 3711.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3826.00 | 3841.28 | 3795.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 3795.00 | 3841.28 | 3795.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 3792.40 | 3825.25 | 3795.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:45:00 | 3790.10 | 3825.25 | 3795.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 3781.60 | 3816.52 | 3794.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 3781.60 | 3816.52 | 3794.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 3728.50 | 3777.06 | 3780.64 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 11:15:00 | 3803.10 | 3783.68 | 3783.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 14:15:00 | 3826.90 | 3799.17 | 3790.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 3747.50 | 3789.77 | 3788.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 3742.00 | 3789.77 | 3788.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 219 — SELL (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 10:15:00 | 3765.80 | 3784.98 | 3786.12 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 3845.10 | 3796.42 | 3790.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 3862.00 | 3821.26 | 3807.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 3883.00 | 3890.37 | 3857.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:00:00 | 3959.90 | 3915.85 | 3887.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:30:00 | 3949.90 | 3932.39 | 3905.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 3949.00 | 3930.05 | 3909.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 15:15:00 | 3875.40 | 3899.19 | 3902.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2025-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 15:15:00 | 3875.40 | 3899.19 | 3902.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 3863.90 | 3892.13 | 3898.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 3854.90 | 3851.06 | 3872.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 3854.90 | 3851.06 | 3872.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 3780.50 | 3836.95 | 3864.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 3776.80 | 3836.95 | 3864.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 3865.40 | 3842.64 | 3864.32 | SL hit (close>static) qty=1.00 sl=3865.00 alert=retest2 |

### Cycle 222 — BUY (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 12:15:00 | 3905.00 | 3877.89 | 3877.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 14:15:00 | 3928.90 | 3893.28 | 3884.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 3974.80 | 3990.92 | 3954.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 3974.80 | 3990.92 | 3954.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 4066.50 | 4099.61 | 4065.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 4066.50 | 4099.61 | 4065.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 4060.00 | 4091.69 | 4065.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 4042.90 | 4091.69 | 4065.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 4055.30 | 4084.41 | 4064.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 4050.00 | 4084.41 | 4064.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 4056.90 | 4078.91 | 4063.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 10:45:00 | 4055.00 | 4078.91 | 4063.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 4066.40 | 4076.41 | 4064.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 4075.00 | 4076.41 | 4064.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 15:15:00 | 4030.10 | 4067.86 | 4064.21 | SL hit (close<static) qty=1.00 sl=4048.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 4028.30 | 4059.95 | 4060.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 10:15:00 | 4001.40 | 4048.24 | 4055.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 4111.20 | 4031.69 | 4039.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 4130.00 | 4031.69 | 4039.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 10:15:00 | 4097.90 | 4044.93 | 4044.50 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 4024.10 | 4048.84 | 4051.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 14:15:00 | 3952.80 | 4019.70 | 4036.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 3941.30 | 3937.07 | 3976.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 3941.30 | 3937.07 | 3976.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 3990.00 | 3947.66 | 3978.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:30:00 | 3930.30 | 3937.37 | 3970.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 3929.50 | 3932.11 | 3962.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 3926.10 | 3912.22 | 3937.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:00:00 | 3931.00 | 3918.74 | 3934.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 3941.40 | 3923.27 | 3935.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 3935.00 | 3923.27 | 3935.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 3950.10 | 3928.64 | 3936.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 3911.90 | 3928.64 | 3936.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 14:15:00 | 3959.50 | 3918.71 | 3926.21 | SL hit (close>static) qty=1.00 sl=3959.00 alert=retest2 |

### Cycle 226 — BUY (started 2026-01-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 12:15:00 | 3798.80 | 3769.61 | 3767.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 15:15:00 | 3925.40 | 3809.01 | 3786.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-28 09:15:00 | 3806.50 | 3808.51 | 3788.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 3806.50 | 3808.51 | 3788.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 3875.00 | 3869.98 | 3846.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 14:45:00 | 3852.00 | 3869.98 | 3846.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 4001.00 | 4001.77 | 3943.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 3917.60 | 4001.77 | 3943.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3915.90 | 3984.59 | 3941.38 | EMA400 retest candle locked (from upside) |

### Cycle 227 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 3839.60 | 3913.65 | 3917.95 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 14:15:00 | 3989.00 | 3928.72 | 3924.41 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 3875.00 | 3918.07 | 3921.19 | EMA200 below EMA400 |

### Cycle 230 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 4024.00 | 3937.13 | 3927.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4050.20 | 3971.08 | 3945.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 3929.20 | 4006.80 | 3984.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 3932.80 | 4006.80 | 3984.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 3941.50 | 3993.74 | 3980.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 3941.50 | 3993.74 | 3980.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 4022.20 | 4037.51 | 4015.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 4022.20 | 4037.51 | 4015.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 3997.10 | 4029.43 | 4013.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:30:00 | 3999.10 | 4029.43 | 4013.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3987.10 | 4020.96 | 4011.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 3987.10 | 4020.96 | 4011.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 3922.70 | 3992.84 | 3999.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 3875.00 | 3953.15 | 3979.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 3916.30 | 3908.69 | 3943.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:00:00 | 3916.30 | 3908.69 | 3943.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 3925.80 | 3912.11 | 3941.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 3925.80 | 3912.11 | 3941.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 3937.70 | 3917.23 | 3941.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 3938.50 | 3917.23 | 3941.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 3962.50 | 3926.28 | 3943.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 3966.20 | 3926.28 | 3943.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 4019.00 | 3944.83 | 3950.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:45:00 | 4025.00 | 3944.83 | 3950.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 232 — BUY (started 2026-02-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 14:15:00 | 4029.90 | 3961.84 | 3957.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 4120.60 | 4003.70 | 3977.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 4025.20 | 4064.80 | 4030.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 4025.20 | 4064.80 | 4030.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 4031.20 | 4058.08 | 4030.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:15:00 | 4025.30 | 4058.08 | 4030.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 4025.60 | 4051.58 | 4030.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:45:00 | 4022.50 | 4051.58 | 4030.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 4017.30 | 4044.73 | 4029.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 4012.00 | 4044.73 | 4029.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 4017.20 | 4039.22 | 4028.00 | EMA400 retest candle locked (from upside) |

### Cycle 233 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 3978.60 | 4020.47 | 4021.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 3939.70 | 3994.27 | 4008.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 3965.00 | 3960.63 | 3982.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:45:00 | 3964.30 | 3960.63 | 3982.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 3868.00 | 3871.88 | 3905.06 | EMA400 retest candle locked (from downside) |

### Cycle 234 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 3945.50 | 3916.75 | 3915.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 15:15:00 | 3963.90 | 3926.18 | 3920.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 3955.30 | 3966.70 | 3950.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 3955.30 | 3966.70 | 3950.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 3951.70 | 3963.70 | 3950.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 3955.90 | 3963.70 | 3950.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 3965.80 | 3964.12 | 3951.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 3976.10 | 3959.34 | 3952.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:30:00 | 3977.90 | 3963.46 | 3955.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 3938.50 | 3958.44 | 3954.49 | SL hit (close<static) qty=1.00 sl=3941.10 alert=retest2 |

### Cycle 235 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 3902.50 | 3943.35 | 3948.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 11:15:00 | 3895.00 | 3928.00 | 3939.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 13:15:00 | 3873.50 | 3867.49 | 3892.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 14:00:00 | 3873.50 | 3867.49 | 3892.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 3911.40 | 3857.61 | 3870.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 3911.40 | 3857.61 | 3870.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 3909.30 | 3867.95 | 3873.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 3865.00 | 3867.95 | 3873.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 09:15:00 | 3926.20 | 3879.60 | 3878.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 236 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 3926.20 | 3879.60 | 3878.44 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3845.60 | 3883.35 | 3887.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 3817.00 | 3859.63 | 3873.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 09:15:00 | 3649.60 | 3633.05 | 3676.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:00:00 | 3649.60 | 3633.05 | 3676.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 3666.00 | 3642.91 | 3673.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:30:00 | 3675.60 | 3642.91 | 3673.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 3672.10 | 3648.75 | 3673.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 3672.10 | 3648.75 | 3673.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 3681.80 | 3655.36 | 3674.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 3681.80 | 3655.36 | 3674.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 3675.00 | 3659.29 | 3674.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:30:00 | 3677.40 | 3659.29 | 3674.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3707.50 | 3625.78 | 3637.90 | EMA400 retest candle locked (from downside) |

### Cycle 238 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 3713.60 | 3656.42 | 3650.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3784.30 | 3699.25 | 3673.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 3711.50 | 3717.79 | 3689.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 3720.00 | 3717.79 | 3689.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3710.00 | 3715.06 | 3693.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 3710.00 | 3715.06 | 3693.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3669.70 | 3714.04 | 3701.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:30:00 | 3708.30 | 3715.35 | 3703.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 3707.80 | 3713.84 | 3703.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 3718.40 | 3711.83 | 3703.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 3712.90 | 3711.75 | 3704.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 3678.00 | 3705.00 | 3702.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 3696.20 | 3705.00 | 3702.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 3661.20 | 3696.24 | 3698.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 239 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 3661.20 | 3696.24 | 3698.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 3648.90 | 3686.77 | 3694.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 3690.10 | 3687.44 | 3693.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 3695.50 | 3687.44 | 3693.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 3672.50 | 3684.45 | 3691.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 12:45:00 | 3689.60 | 3684.45 | 3691.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 3668.60 | 3680.57 | 3688.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 3687.40 | 3680.57 | 3688.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3665.50 | 3675.06 | 3684.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 3686.90 | 3675.06 | 3684.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3662.60 | 3654.67 | 3669.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3676.80 | 3654.67 | 3669.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 3698.40 | 3663.41 | 3672.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 3825.20 | 3663.41 | 3672.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 240 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 3900.00 | 3710.73 | 3693.04 | EMA200 above EMA400 |

### Cycle 241 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 3644.20 | 3713.70 | 3720.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3585.10 | 3653.56 | 3686.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3335.70 | 3331.28 | 3407.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 3335.70 | 3331.28 | 3407.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3369.00 | 3350.54 | 3403.66 | EMA400 retest candle locked (from downside) |

### Cycle 242 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 3505.80 | 3430.65 | 3423.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 12:15:00 | 3601.80 | 3517.61 | 3484.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 3635.00 | 3638.33 | 3584.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 3584.10 | 3638.33 | 3584.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3590.90 | 3628.85 | 3585.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:30:00 | 3633.20 | 3627.78 | 3592.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 3662.20 | 3650.18 | 3612.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 3659.80 | 3646.09 | 3620.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 3633.50 | 3667.06 | 3668.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 243 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 3633.50 | 3667.06 | 3668.59 | EMA200 below EMA400 |

### Cycle 244 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 3780.90 | 3685.50 | 3676.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 10:15:00 | 3824.10 | 3766.10 | 3729.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 14:15:00 | 3925.00 | 3929.92 | 3885.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 14:30:00 | 3922.00 | 3929.92 | 3885.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 3990.90 | 3941.26 | 3897.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 4047.70 | 3978.14 | 3938.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 12:00:00 | 4025.00 | 3993.71 | 3952.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:00:00 | 4008.70 | 3996.71 | 3957.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 4066.40 | 3998.97 | 3968.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4002.60 | 4011.30 | 3993.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 13:30:00 | 4033.30 | 4015.57 | 4000.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 4030.50 | 4015.57 | 4000.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 4024.60 | 4038.44 | 4018.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 12:00:00 | 4028.90 | 4047.48 | 4036.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 4020.00 | 4041.99 | 4034.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-24 15:15:00 | 4000.70 | 4025.73 | 4028.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 245 — SELL (started 2026-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 15:15:00 | 4000.70 | 4025.73 | 4028.52 | EMA200 below EMA400 |

### Cycle 246 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 4085.50 | 4037.69 | 4033.70 | EMA200 above EMA400 |

### Cycle 247 — SELL (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 13:15:00 | 4011.10 | 4028.40 | 4030.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 13:15:00 | 3983.60 | 4007.04 | 4017.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 4017.50 | 4003.56 | 4012.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:30:00 | 4006.00 | 4003.56 | 4012.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 4015.90 | 4006.03 | 4012.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 10:30:00 | 4017.00 | 4006.03 | 4012.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 3989.10 | 4002.64 | 4010.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:15:00 | 3964.50 | 3999.79 | 4008.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 3981.60 | 3949.69 | 3963.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:30:00 | 3976.10 | 3951.98 | 3962.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 3954.10 | 3951.98 | 3962.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 3940.40 | 3949.66 | 3960.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 3924.80 | 3949.66 | 3960.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 3974.40 | 3955.44 | 3960.43 | SL hit (close>static) qty=1.00 sl=3963.60 alert=retest2 |

### Cycle 248 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 3985.10 | 3926.89 | 3923.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 11:15:00 | 4001.50 | 3941.81 | 3930.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 3987.90 | 3996.48 | 3974.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 3987.90 | 3996.48 | 3974.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 3972.50 | 3991.68 | 3974.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 3972.50 | 3991.68 | 3974.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 3955.00 | 3984.35 | 3972.89 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-22 09:15:00 | 2860.55 | 2023-05-25 14:15:00 | 2877.55 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2023-06-02 11:30:00 | 3044.85 | 2023-06-15 09:15:00 | 3349.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-02 12:15:00 | 3049.90 | 2023-06-15 09:15:00 | 3354.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-05 09:30:00 | 3047.00 | 2023-06-15 09:15:00 | 3351.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-05 10:15:00 | 3049.00 | 2023-06-15 09:15:00 | 3353.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-09 09:15:00 | 3162.00 | 2023-06-16 11:15:00 | 3478.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2023-06-21 14:15:00 | 3350.25 | 2023-06-28 09:15:00 | 3323.85 | STOP_HIT | 1.00 | 0.79% |
| SELL | retest2 | 2023-06-22 09:45:00 | 3348.00 | 2023-06-30 13:15:00 | 3182.74 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2023-06-22 12:00:00 | 3343.50 | 2023-06-30 13:15:00 | 3180.60 | PARTIAL | 0.50 | 4.87% |
| SELL | retest2 | 2023-06-26 09:45:00 | 3331.95 | 2023-07-03 09:15:00 | 3176.32 | PARTIAL | 0.50 | 4.67% |
| SELL | retest2 | 2023-06-26 14:30:00 | 3309.95 | 2023-07-03 09:15:00 | 3165.35 | PARTIAL | 0.50 | 4.37% |
| SELL | retest2 | 2023-06-26 15:15:00 | 3305.00 | 2023-07-03 12:15:00 | 3144.45 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2023-06-27 09:45:00 | 3314.95 | 2023-07-03 12:15:00 | 3149.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-27 10:15:00 | 3312.25 | 2023-07-03 12:15:00 | 3146.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-28 09:15:00 | 3281.10 | 2023-07-03 13:15:00 | 3139.75 | PARTIAL | 0.50 | 4.31% |
| SELL | retest2 | 2023-06-28 11:30:00 | 3289.95 | 2023-07-04 12:15:00 | 3125.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-22 09:45:00 | 3348.00 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 5.59% |
| SELL | retest2 | 2023-06-22 12:00:00 | 3343.50 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 5.46% |
| SELL | retest2 | 2023-06-26 09:45:00 | 3331.95 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2023-06-26 14:30:00 | 3309.95 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2023-06-26 15:15:00 | 3305.00 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2023-06-27 09:45:00 | 3314.95 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2023-06-27 10:15:00 | 3312.25 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 4.57% |
| SELL | retest2 | 2023-06-28 09:15:00 | 3281.10 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2023-06-28 11:30:00 | 3289.95 | 2023-07-04 15:15:00 | 3161.00 | STOP_HIT | 0.50 | 3.92% |
| BUY | retest2 | 2023-07-18 09:15:00 | 3393.30 | 2023-07-25 12:15:00 | 3498.00 | STOP_HIT | 1.00 | 3.09% |
| BUY | retest2 | 2023-07-18 10:00:00 | 3393.90 | 2023-07-25 12:15:00 | 3498.00 | STOP_HIT | 1.00 | 3.07% |
| SELL | retest2 | 2023-07-27 11:30:00 | 3489.95 | 2023-07-28 11:15:00 | 3536.05 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2023-07-27 12:15:00 | 3483.30 | 2023-07-28 11:15:00 | 3536.05 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2023-07-27 14:30:00 | 3478.75 | 2023-07-28 11:15:00 | 3536.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2023-07-28 09:15:00 | 3478.75 | 2023-07-28 11:15:00 | 3536.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2023-08-11 09:15:00 | 3616.15 | 2023-08-17 10:15:00 | 3584.65 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-08-14 09:30:00 | 3599.00 | 2023-08-17 10:15:00 | 3584.65 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-08-14 10:30:00 | 3606.00 | 2023-08-17 10:15:00 | 3584.65 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-08-17 10:15:00 | 3595.75 | 2023-08-17 10:15:00 | 3584.65 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2023-08-23 09:45:00 | 3646.75 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2023-08-23 14:30:00 | 3643.95 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | 1.02% |
| BUY | retest2 | 2023-08-24 14:45:00 | 3640.05 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2023-08-25 09:15:00 | 3660.00 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2023-08-28 09:15:00 | 3707.00 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-09-01 09:15:00 | 3721.20 | 2023-09-04 11:15:00 | 3681.15 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-09-06 09:15:00 | 3666.00 | 2023-09-08 10:15:00 | 3665.85 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2023-09-14 12:15:00 | 3610.00 | 2023-09-15 14:15:00 | 3686.15 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2023-09-22 10:15:00 | 3492.70 | 2023-09-28 09:15:00 | 3503.80 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-09-22 13:00:00 | 3500.05 | 2023-09-28 09:15:00 | 3503.80 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2023-10-09 09:15:00 | 3415.00 | 2023-10-10 13:15:00 | 3444.30 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-10-12 09:45:00 | 3400.85 | 2023-10-16 13:15:00 | 3430.25 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2023-10-20 10:30:00 | 3575.00 | 2023-10-20 12:15:00 | 3471.15 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2023-10-26 09:15:00 | 3366.55 | 2023-10-27 10:15:00 | 3486.40 | STOP_HIT | 1.00 | -3.56% |
| BUY | retest2 | 2023-10-30 15:00:00 | 3487.45 | 2023-11-07 09:15:00 | 3578.85 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2023-10-31 09:15:00 | 3484.60 | 2023-11-07 09:15:00 | 3578.85 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2023-11-09 10:15:00 | 3562.00 | 2023-11-10 10:15:00 | 3629.50 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-11-29 10:15:00 | 3580.00 | 2023-12-11 12:15:00 | 3694.00 | STOP_HIT | 1.00 | 3.18% |
| BUY | retest2 | 2023-11-29 10:45:00 | 3581.90 | 2023-12-11 12:15:00 | 3694.00 | STOP_HIT | 1.00 | 3.13% |
| SELL | retest2 | 2023-12-18 09:15:00 | 3587.25 | 2023-12-22 14:15:00 | 3574.90 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2023-12-18 10:00:00 | 3577.10 | 2023-12-22 14:15:00 | 3574.90 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest1 | 2023-12-29 10:30:00 | 3689.25 | 2024-01-02 09:15:00 | 3612.50 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-01-01 12:15:00 | 3692.75 | 2024-01-02 09:15:00 | 3612.50 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2024-01-01 14:00:00 | 3692.30 | 2024-01-02 09:15:00 | 3612.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-01-01 15:15:00 | 3690.00 | 2024-01-02 09:15:00 | 3612.50 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-01-04 12:30:00 | 3588.95 | 2024-01-05 11:15:00 | 3621.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-01-05 10:30:00 | 3588.90 | 2024-01-05 11:15:00 | 3621.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-01-08 09:45:00 | 3596.45 | 2024-01-08 15:15:00 | 3624.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-01-08 12:15:00 | 3603.25 | 2024-01-08 15:15:00 | 3624.00 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-01-10 10:15:00 | 3823.00 | 2024-01-15 14:15:00 | 3704.65 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2024-01-10 11:15:00 | 3768.45 | 2024-01-15 14:15:00 | 3704.65 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-01-11 11:00:00 | 3774.25 | 2024-01-15 14:15:00 | 3704.65 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2024-01-12 11:45:00 | 3768.60 | 2024-01-16 11:15:00 | 3709.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2024-01-15 09:30:00 | 3755.70 | 2024-01-16 11:15:00 | 3709.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-01-15 11:15:00 | 3755.20 | 2024-01-16 11:15:00 | 3709.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-01-15 13:15:00 | 3755.00 | 2024-01-16 11:15:00 | 3709.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-01-17 09:15:00 | 3678.80 | 2024-01-17 13:15:00 | 3730.55 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-01-31 09:15:00 | 3994.75 | 2024-02-01 13:15:00 | 4394.23 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-02-14 09:15:00 | 3942.35 | 2024-02-19 13:15:00 | 3952.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-02-26 09:15:00 | 3726.20 | 2024-02-27 10:15:00 | 3759.95 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-03-11 09:15:00 | 3672.75 | 2024-03-12 15:15:00 | 3709.95 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-03-11 13:00:00 | 3666.00 | 2024-03-12 15:15:00 | 3709.95 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-03-15 10:15:00 | 3575.50 | 2024-03-19 13:15:00 | 3637.35 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-03-18 09:15:00 | 3582.80 | 2024-03-19 13:15:00 | 3637.35 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-03-20 12:30:00 | 3656.55 | 2024-04-02 09:15:00 | 4022.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-20 15:00:00 | 3649.70 | 2024-04-02 09:15:00 | 4014.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-03-21 09:15:00 | 3668.15 | 2024-04-02 09:15:00 | 4034.97 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-08 09:15:00 | 4075.35 | 2024-04-10 14:15:00 | 3999.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-04-18 14:15:00 | 3827.00 | 2024-04-24 09:15:00 | 3919.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-04-19 14:45:00 | 3825.20 | 2024-04-24 09:15:00 | 3919.90 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2024-04-23 09:30:00 | 3827.05 | 2024-04-24 09:15:00 | 3919.90 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-04-23 10:30:00 | 3824.55 | 2024-04-24 09:15:00 | 3919.90 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-05-06 13:15:00 | 3764.50 | 2024-05-07 09:15:00 | 3852.15 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-05-06 14:30:00 | 3761.95 | 2024-05-07 09:15:00 | 3852.15 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2024-05-13 10:30:00 | 3723.30 | 2024-05-14 11:15:00 | 3792.65 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-05-27 11:45:00 | 3719.80 | 2024-05-28 13:15:00 | 3700.00 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-05-27 13:15:00 | 3717.15 | 2024-05-28 13:15:00 | 3700.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-05-28 10:45:00 | 3708.50 | 2024-05-28 13:15:00 | 3700.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-05-28 13:15:00 | 3707.45 | 2024-05-28 13:15:00 | 3700.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-06-12 12:30:00 | 3843.00 | 2024-06-19 10:15:00 | 4227.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 15:15:00 | 3834.00 | 2024-06-19 10:15:00 | 4217.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 10:15:00 | 3832.05 | 2024-06-19 10:15:00 | 4215.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-13 11:00:00 | 3832.95 | 2024-06-19 10:15:00 | 4216.24 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-21 12:15:00 | 4227.50 | 2024-06-25 15:15:00 | 4212.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-06-21 15:15:00 | 4235.00 | 2024-06-25 15:15:00 | 4212.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-06-25 13:00:00 | 4227.75 | 2024-06-25 15:15:00 | 4212.00 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-06-27 10:30:00 | 4170.15 | 2024-07-01 10:15:00 | 4237.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-06-28 13:15:00 | 4184.10 | 2024-07-01 10:15:00 | 4237.10 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-07-05 15:15:00 | 4274.00 | 2024-07-08 09:15:00 | 4204.65 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-07-18 13:00:00 | 4205.15 | 2024-07-19 09:15:00 | 4309.95 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-07-22 09:30:00 | 4371.45 | 2024-07-23 14:15:00 | 4309.05 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-07-22 12:00:00 | 4393.20 | 2024-07-23 14:15:00 | 4309.05 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2024-07-22 13:15:00 | 4367.95 | 2024-07-23 14:15:00 | 4309.05 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-08-08 09:15:00 | 4584.00 | 2024-08-13 11:15:00 | 4551.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-08-26 10:15:00 | 4428.50 | 2024-09-03 09:15:00 | 4392.55 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2024-08-26 11:45:00 | 4420.75 | 2024-09-03 09:15:00 | 4392.55 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2024-08-27 11:00:00 | 4426.50 | 2024-09-03 10:15:00 | 4396.05 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2024-08-28 09:30:00 | 4387.00 | 2024-09-03 10:15:00 | 4396.05 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-08-30 13:00:00 | 4323.10 | 2024-09-03 10:15:00 | 4396.05 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2024-09-02 10:30:00 | 4328.95 | 2024-09-03 10:15:00 | 4396.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-09-25 15:00:00 | 4335.45 | 2024-09-26 09:15:00 | 4311.25 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-10-07 10:30:00 | 4160.00 | 2024-10-15 11:15:00 | 4146.70 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-10-09 13:00:00 | 4170.35 | 2024-10-15 11:15:00 | 4146.70 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-10-28 09:15:00 | 3868.80 | 2024-10-30 11:15:00 | 3929.70 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-10-28 11:45:00 | 3873.20 | 2024-10-30 11:15:00 | 3929.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-10-28 15:00:00 | 3877.85 | 2024-10-30 11:15:00 | 3929.70 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-10-29 09:30:00 | 3860.00 | 2024-10-30 11:15:00 | 3929.70 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2024-10-30 09:30:00 | 3874.60 | 2024-10-30 11:15:00 | 3929.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest1 | 2024-11-04 09:15:00 | 3796.60 | 2024-11-08 09:15:00 | 3606.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-05 09:15:00 | 3799.05 | 2024-11-08 09:15:00 | 3609.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-04 09:15:00 | 3796.60 | 2024-11-14 15:15:00 | 3419.15 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest1 | 2024-11-05 09:15:00 | 3799.05 | 2024-11-18 09:15:00 | 3416.94 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2024-11-21 10:30:00 | 3362.75 | 2024-11-25 09:15:00 | 3492.75 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2024-11-21 11:30:00 | 3360.20 | 2024-11-25 09:15:00 | 3492.75 | STOP_HIT | 1.00 | -3.94% |
| BUY | retest2 | 2024-12-03 09:15:00 | 3495.80 | 2024-12-05 09:15:00 | 3463.40 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-12-09 13:30:00 | 3455.85 | 2024-12-10 09:15:00 | 3512.40 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-12-09 15:00:00 | 3455.00 | 2024-12-10 09:15:00 | 3512.40 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-12-16 15:00:00 | 3505.30 | 2024-12-17 12:15:00 | 3490.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-26 13:15:00 | 3343.10 | 2024-12-27 09:15:00 | 3401.85 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest1 | 2025-01-20 09:45:00 | 3530.80 | 2025-01-21 10:15:00 | 3489.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest1 | 2025-01-20 10:15:00 | 3523.50 | 2025-01-21 10:15:00 | 3489.50 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest1 | 2025-01-20 12:00:00 | 3530.05 | 2025-01-21 10:15:00 | 3489.50 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-01-22 11:15:00 | 3583.60 | 2025-01-30 13:15:00 | 3625.00 | STOP_HIT | 1.00 | 1.16% |
| BUY | retest2 | 2025-02-01 15:15:00 | 3693.00 | 2025-02-03 14:15:00 | 3650.80 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-02-06 14:30:00 | 3706.95 | 2025-02-06 15:15:00 | 3650.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-02-13 10:30:00 | 3523.50 | 2025-02-17 09:15:00 | 3347.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 12:15:00 | 3527.90 | 2025-02-17 09:15:00 | 3351.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-02-13 10:30:00 | 3523.50 | 2025-02-17 12:15:00 | 3445.50 | STOP_HIT | 0.50 | 2.21% |
| SELL | retest1 | 2025-02-13 12:15:00 | 3527.90 | 2025-02-17 12:15:00 | 3445.50 | STOP_HIT | 0.50 | 2.34% |
| SELL | retest2 | 2025-02-14 10:30:00 | 3497.40 | 2025-02-24 09:15:00 | 3322.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:30:00 | 3497.40 | 2025-02-28 09:15:00 | 3147.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-27 11:15:00 | 3352.00 | 2025-03-28 13:15:00 | 3333.20 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-03-27 14:15:00 | 3362.70 | 2025-03-28 13:15:00 | 3333.20 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-03-28 09:15:00 | 3404.70 | 2025-03-28 13:15:00 | 3333.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-04-04 09:45:00 | 3268.00 | 2025-04-07 09:15:00 | 3104.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-04 09:45:00 | 3268.00 | 2025-04-08 13:15:00 | 3112.65 | STOP_HIT | 0.50 | 4.75% |
| BUY | retest2 | 2025-04-16 09:45:00 | 3186.60 | 2025-04-17 09:15:00 | 3151.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-04-16 14:45:00 | 3187.20 | 2025-04-17 09:15:00 | 3151.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-05-08 12:45:00 | 3128.50 | 2025-05-12 09:15:00 | 3180.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-05-15 10:30:00 | 3303.60 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-05-15 12:30:00 | 3303.10 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-05-19 09:45:00 | 3310.50 | 2025-05-20 15:15:00 | 3287.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest1 | 2025-06-16 09:15:00 | 3406.40 | 2025-06-18 09:15:00 | 3441.70 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3400.80 | 2025-06-20 14:15:00 | 3230.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 12:15:00 | 3404.40 | 2025-06-20 14:15:00 | 3234.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-19 09:30:00 | 3401.20 | 2025-06-20 14:15:00 | 3231.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-18 11:15:00 | 3400.80 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.05% |
| SELL | retest2 | 2025-06-18 12:15:00 | 3404.40 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2025-06-19 09:30:00 | 3401.20 | 2025-06-25 09:15:00 | 3297.10 | STOP_HIT | 0.50 | 3.06% |
| BUY | retest2 | 2025-07-10 14:45:00 | 3438.70 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-11 09:15:00 | 3433.90 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-07-11 09:45:00 | 3442.00 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-07-24 10:30:00 | 3336.10 | 2025-07-29 10:15:00 | 3169.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 3336.10 | 2025-07-30 11:15:00 | 3200.10 | STOP_HIT | 0.50 | 4.08% |
| SELL | retest2 | 2025-08-12 10:15:00 | 3093.10 | 2025-08-13 10:15:00 | 3124.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-25 13:15:00 | 3091.60 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.19% |
| SELL | retest2 | 2025-08-25 14:15:00 | 3094.20 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2025-08-25 15:00:00 | 3091.20 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-08-26 09:30:00 | 3089.40 | 2025-09-01 10:15:00 | 3085.70 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2025-09-18 09:30:00 | 3131.00 | 2025-09-19 14:15:00 | 3069.40 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-09-19 11:15:00 | 3128.50 | 2025-09-19 14:15:00 | 3069.40 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-09-24 11:15:00 | 3059.20 | 2025-09-25 10:15:00 | 3086.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-07 11:30:00 | 3194.30 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-10-07 12:45:00 | 3195.10 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2025-10-08 10:15:00 | 3189.00 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2025-10-08 11:30:00 | 3198.00 | 2025-10-13 13:15:00 | 3215.50 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2025-10-15 12:15:00 | 3214.70 | 2025-10-15 15:15:00 | 3270.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2025-10-20 09:30:00 | 3323.10 | 2025-10-20 10:15:00 | 3292.90 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-10-20 13:15:00 | 3320.40 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-10-20 14:00:00 | 3320.20 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2025-10-21 13:45:00 | 3328.10 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-10-23 10:15:00 | 3347.20 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-27 09:15:00 | 3371.80 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-10-27 11:15:00 | 3380.70 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-10-27 12:30:00 | 3367.60 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-10-27 14:00:00 | 3371.00 | 2025-10-28 14:15:00 | 3330.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-29 14:45:00 | 3328.90 | 2025-11-04 11:15:00 | 3315.00 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3316.40 | 2025-11-04 11:15:00 | 3315.00 | STOP_HIT | 1.00 | 0.04% |
| BUY | retest2 | 2025-11-06 14:00:00 | 3314.90 | 2025-11-06 15:15:00 | 3285.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-11-18 12:15:00 | 3731.70 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-11-19 10:15:00 | 3716.00 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-19 15:00:00 | 3724.60 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-20 10:00:00 | 3725.00 | 2025-11-21 14:15:00 | 3703.90 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-11-24 14:45:00 | 3700.00 | 2025-11-25 09:15:00 | 3732.20 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-24 15:15:00 | 3701.00 | 2025-11-25 09:15:00 | 3732.20 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-27 09:15:00 | 3779.50 | 2025-12-04 15:15:00 | 3810.50 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-12-24 10:00:00 | 3959.90 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-12-24 13:30:00 | 3949.90 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-26 09:15:00 | 3949.00 | 2025-12-26 15:15:00 | 3875.40 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-12-30 09:15:00 | 3776.80 | 2025-12-30 09:15:00 | 3865.40 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2026-01-06 12:15:00 | 4075.00 | 2026-01-06 15:15:00 | 4030.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-13 09:30:00 | 3930.30 | 2026-01-16 14:15:00 | 3959.50 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3929.50 | 2026-01-20 14:15:00 | 3733.78 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2026-01-14 11:00:00 | 3926.10 | 2026-01-20 14:15:00 | 3733.02 | PARTIAL | 0.50 | 4.92% |
| SELL | retest2 | 2026-01-14 14:00:00 | 3931.00 | 2026-01-20 14:15:00 | 3729.79 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2026-01-16 09:15:00 | 3911.90 | 2026-01-20 14:15:00 | 3734.45 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3902.20 | 2026-01-21 10:15:00 | 3707.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3929.50 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.92% |
| SELL | retest2 | 2026-01-14 11:00:00 | 3926.10 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.84% |
| SELL | retest2 | 2026-01-14 14:00:00 | 3931.00 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.96% |
| SELL | retest2 | 2026-01-16 09:15:00 | 3911.90 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.49% |
| SELL | retest2 | 2026-01-19 09:15:00 | 3902.20 | 2026-01-22 09:15:00 | 3814.60 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest2 | 2026-02-20 09:30:00 | 3976.10 | 2026-02-20 12:15:00 | 3938.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-20 10:30:00 | 3977.90 | 2026-02-20 12:15:00 | 3938.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-02-26 09:15:00 | 3865.00 | 2026-02-26 09:15:00 | 3926.20 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-03-12 10:30:00 | 3708.30 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2026-03-12 12:00:00 | 3707.80 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-03-12 13:15:00 | 3718.40 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-12 15:00:00 | 3712.90 | 2026-03-13 09:15:00 | 3661.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-04-02 11:30:00 | 3633.20 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-04-02 14:45:00 | 3662.20 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2026-04-06 11:15:00 | 3659.80 | 2026-04-09 14:15:00 | 3633.50 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-20 09:45:00 | 4047.70 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-04-20 12:00:00 | 4025.00 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-04-20 13:00:00 | 4008.70 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-04-21 09:15:00 | 4066.40 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-22 13:30:00 | 4033.30 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2026-04-22 14:00:00 | 4030.50 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-04-23 10:30:00 | 4024.60 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2026-04-24 12:00:00 | 4028.90 | 2026-04-24 15:15:00 | 4000.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2026-04-29 13:15:00 | 3964.50 | 2026-05-04 15:15:00 | 3974.40 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2026-05-04 10:00:00 | 3981.60 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2026-05-04 11:30:00 | 3976.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-05-04 12:00:00 | 3954.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-05-04 13:15:00 | 3924.80 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-05-05 10:00:00 | 3923.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-05-05 10:30:00 | 3930.10 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-05-06 14:00:00 | 3919.80 | 2026-05-07 10:15:00 | 3985.10 | STOP_HIT | 1.00 | -1.67% |
