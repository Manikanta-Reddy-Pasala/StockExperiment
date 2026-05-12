# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3032.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 14 |
| ALERT2 | 13 |
| ALERT2_SKIP | 6 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 3 |
| TARGET_HIT | 6 |
| STOP_HIT | 33 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 33
- **Target hits / Stop hits / Partials:** 6 / 33 / 3
- **Avg / median % per leg:** 0.56% / -1.00%
- **Sum % (uncompounded):** 23.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 32 | 3 | 9.4% | 3 | 29 | 0 | -0.48% | -15.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 32 | 3 | 9.4% | 3 | 29 | 0 | -0.48% | -15.3% |
| SELL (all) | 10 | 6 | 60.0% | 3 | 4 | 3 | 3.88% | 38.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 10 | 6 | 60.0% | 3 | 4 | 3 | 3.88% | 38.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 9 | 21.4% | 6 | 33 | 3 | 0.56% | 23.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 09:15:00 | 2868.00 | 2943.50 | 2943.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 14:15:00 | 2864.20 | 2939.74 | 2941.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 2937.00 | 2929.77 | 2936.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-01 11:15:00 | 2937.00 | 2929.77 | 2936.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 11:15:00 | 2937.00 | 2929.77 | 2936.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:00:00 | 2937.00 | 2929.77 | 2936.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 12:15:00 | 2980.30 | 2930.28 | 2936.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 12:45:00 | 2967.95 | 2930.28 | 2936.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 2979.00 | 2930.76 | 2936.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:15:00 | 2990.00 | 2930.76 | 2936.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 11:15:00 | 3160.00 | 2943.86 | 2943.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 3160.20 | 2948.12 | 2945.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-27 09:15:00 | 3131.05 | 3162.52 | 3081.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 10:00:00 | 3131.05 | 3162.52 | 3081.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 09:15:00 | 3100.00 | 3159.38 | 3082.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 09:45:00 | 3080.00 | 3159.38 | 3082.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 3107.25 | 3157.67 | 3083.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:30:00 | 3095.00 | 3157.67 | 3083.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 3098.95 | 3157.08 | 3083.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 09:15:00 | 2969.65 | 3157.08 | 3083.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 09:15:00 | 3011.05 | 3155.63 | 3083.46 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 11:15:00 | 3002.00 | 3039.51 | 3039.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 2954.90 | 3037.99 | 3038.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 3032.10 | 3018.03 | 3028.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 3032.10 | 3018.03 | 3028.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 3032.10 | 3018.03 | 3028.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:45:00 | 3038.60 | 3018.03 | 3028.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 3057.90 | 3018.43 | 3028.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 11:00:00 | 3057.90 | 3018.43 | 3028.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 3055.10 | 3018.79 | 3028.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:15:00 | 3055.25 | 3018.79 | 3028.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 10:15:00 | 3053.00 | 3020.88 | 3029.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 10:45:00 | 3050.85 | 3020.88 | 3029.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 11:15:00 | 3047.00 | 3021.14 | 3029.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 11:30:00 | 3056.35 | 3021.14 | 3029.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 13:15:00 | 3057.60 | 3021.74 | 3029.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 14:00:00 | 3057.60 | 3021.74 | 3029.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 15:15:00 | 3060.00 | 3022.49 | 3029.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-31 09:15:00 | 3135.05 | 3022.49 | 3029.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-11-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 13:15:00 | 3176.50 | 3037.58 | 3037.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-03 09:15:00 | 3196.85 | 3046.38 | 3041.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 12:15:00 | 3506.00 | 3508.90 | 3382.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-03 12:45:00 | 3505.25 | 3508.90 | 3382.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 15:15:00 | 3775.00 | 3791.38 | 3630.32 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 13:15:00 | 3500.15 | 3588.08 | 3588.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-18 15:15:00 | 3498.00 | 3586.39 | 3587.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 09:15:00 | 3648.90 | 3577.39 | 3582.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 3648.90 | 3577.39 | 3582.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 3648.90 | 3577.39 | 3582.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:00:00 | 3648.90 | 3577.39 | 3582.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 10:15:00 | 3629.30 | 3577.91 | 3582.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-21 10:30:00 | 3637.90 | 3577.91 | 3582.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 3748.55 | 3588.47 | 3587.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 09:15:00 | 3775.05 | 3594.70 | 3591.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-04 14:15:00 | 3600.00 | 3646.10 | 3620.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 14:15:00 | 3600.00 | 3646.10 | 3620.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 3600.00 | 3646.10 | 3620.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-04 15:00:00 | 3600.00 | 3646.10 | 3620.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 3596.00 | 3645.60 | 3620.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 3640.25 | 3645.60 | 3620.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 11:00:00 | 3600.40 | 3644.70 | 3620.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-05 12:15:00 | 3605.00 | 3644.25 | 3620.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-08 09:15:00 | 3606.25 | 3643.30 | 3620.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 09:15:00 | 3602.10 | 3642.89 | 3620.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:15:00 | 3600.00 | 3642.89 | 3620.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 10:15:00 | 3596.10 | 3642.42 | 3620.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 10:30:00 | 3599.25 | 3642.42 | 3620.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-08 11:15:00 | 3588.80 | 3641.89 | 3620.08 | SL hit (close<static) qty=1.00 sl=3590.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-04-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 12:15:00 | 3528.00 | 3601.97 | 3602.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 3501.00 | 3600.15 | 3601.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-23 11:15:00 | 3600.15 | 3586.13 | 3593.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 11:15:00 | 3600.15 | 3586.13 | 3593.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 11:15:00 | 3600.15 | 3586.13 | 3593.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:00:00 | 3600.15 | 3586.13 | 3593.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 12:15:00 | 3600.00 | 3586.27 | 3593.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-23 12:30:00 | 3600.00 | 3586.27 | 3593.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 15:15:00 | 3590.00 | 3586.44 | 3593.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 09:30:00 | 3592.05 | 3586.51 | 3593.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 10:15:00 | 3602.70 | 3586.67 | 3593.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:45:00 | 3601.00 | 3586.67 | 3593.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 3615.90 | 3586.97 | 3594.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 12:00:00 | 3615.90 | 3586.97 | 3594.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 3607.80 | 3587.45 | 3594.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 15:00:00 | 3607.80 | 3587.45 | 3594.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 3605.00 | 3587.62 | 3594.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 3588.45 | 3587.62 | 3594.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 3596.00 | 3587.67 | 3594.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:00:00 | 3596.00 | 3587.67 | 3594.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 3589.75 | 3587.69 | 3594.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 13:00:00 | 3577.70 | 3587.59 | 3594.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-26 11:15:00 | 3601.05 | 3587.36 | 3593.81 | SL hit (close>static) qty=1.00 sl=3596.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 14:15:00 | 3731.85 | 3600.36 | 3599.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-30 15:15:00 | 3750.00 | 3601.85 | 3600.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 12:15:00 | 3732.00 | 3741.57 | 3684.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:00:00 | 3732.00 | 3741.57 | 3684.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 3709.90 | 3746.08 | 3690.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:30:00 | 3732.60 | 3738.38 | 3690.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:15:00 | 3737.90 | 3738.38 | 3690.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 14:15:00 | 3687.65 | 3753.40 | 3706.04 | SL hit (close<static) qty=1.00 sl=3690.55 alert=retest2 |

### Cycle 9 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 3481.40 | 4168.15 | 4169.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 3449.20 | 4154.31 | 4162.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 3751.00 | 3746.78 | 3875.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 12:45:00 | 3764.75 | 3746.78 | 3875.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2799.00 | 2669.03 | 2806.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 11:00:00 | 2799.00 | 2669.03 | 2806.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 2779.30 | 2670.13 | 2806.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 12:15:00 | 2769.90 | 2670.13 | 2806.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 2857.90 | 2676.63 | 2806.26 | SL hit (close>static) qty=1.00 sl=2808.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 3001.00 | 2876.90 | 2876.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3033.20 | 2879.68 | 2877.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 2978.00 | 2992.28 | 2948.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 2978.00 | 2992.28 | 2948.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2940.00 | 2991.29 | 2949.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:30:00 | 3029.00 | 2950.74 | 2938.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 2892.30 | 2974.20 | 2955.20 | SL hit (close<static) qty=1.00 sl=2895.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 2852.10 | 2941.23 | 2941.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 2836.60 | 2940.19 | 2941.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 2917.50 | 2913.41 | 2926.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:30:00 | 2907.50 | 2913.41 | 2926.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2878.70 | 2911.70 | 2924.73 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 3056.30 | 2935.40 | 2934.81 | EMA200 above EMA400 |
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

### Cycle 13 — SELL (started 2025-11-10 15:15:00)

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

### Cycle 14 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 2738.80 | 2571.21 | 2571.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 2742.20 | 2577.78 | 2574.34 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-05 09:15:00 | 3640.25 | 2024-04-08 11:15:00 | 3588.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-04-05 11:00:00 | 3600.40 | 2024-04-08 11:15:00 | 3588.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-04-05 12:15:00 | 3605.00 | 2024-04-08 11:15:00 | 3588.80 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-04-08 09:15:00 | 3606.25 | 2024-04-08 11:15:00 | 3588.80 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-04-09 12:00:00 | 3562.75 | 2024-04-09 13:15:00 | 3527.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-04-10 15:00:00 | 3568.80 | 2024-04-12 09:15:00 | 3537.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-04-12 12:00:00 | 3562.35 | 2024-04-12 12:15:00 | 3539.85 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-04-25 13:00:00 | 3577.70 | 2024-04-26 11:15:00 | 3601.05 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-04-26 12:45:00 | 3581.25 | 2024-04-26 13:15:00 | 3600.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-04-26 14:15:00 | 3583.50 | 2024-04-26 15:15:00 | 3650.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-05-23 09:30:00 | 3732.60 | 2024-05-29 14:15:00 | 3687.65 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-05-23 10:15:00 | 3737.90 | 2024-05-29 14:15:00 | 3687.65 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-05-31 09:30:00 | 3756.55 | 2024-05-31 13:15:00 | 3690.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2024-05-31 11:30:00 | 3739.05 | 2024-05-31 13:15:00 | 3690.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-06-03 09:15:00 | 3715.00 | 2024-06-10 09:15:00 | 3966.00 | TARGET_HIT | 1.00 | 6.76% |
| BUY | retest2 | 2024-06-05 09:45:00 | 3605.45 | 2024-06-11 14:15:00 | 4086.50 | TARGET_HIT | 1.00 | 13.34% |
| BUY | retest2 | 2024-11-11 13:45:00 | 3606.95 | 2024-11-13 09:15:00 | 3493.50 | STOP_HIT | 1.00 | -3.15% |
| BUY | retest2 | 2024-11-12 09:15:00 | 3690.80 | 2024-11-13 09:15:00 | 3493.50 | STOP_HIT | 1.00 | -5.35% |
| SELL | retest2 | 2025-04-29 12:15:00 | 2769.90 | 2025-04-30 09:15:00 | 2857.90 | STOP_HIT | 1.00 | -3.18% |
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
