# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 3600.00
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
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 47 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 38 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 40 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 39
- **Target hits / Stop hits / Partials:** 2 / 39 / 0
- **Avg / median % per leg:** -1.92% / -1.36%
- **Sum % (uncompounded):** -78.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 2 | 25.0% | 2 | 6 | 0 | 0.08% | 0.7% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| BUY @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 2 | 3 | 0 | 2.91% | 14.6% |
| SELL (all) | 33 | 0 | 0.0% | 0 | 33 | 0 | -2.41% | -79.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 0 | 0.0% | 0 | 33 | 0 | -2.41% | -79.5% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -4.64% | -13.9% |
| retest2 (combined) | 38 | 2 | 5.3% | 2 | 36 | 0 | -1.71% | -65.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3126.90 | 2692.85 | 2691.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3142.70 | 2697.33 | 2693.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 3365.00 | 3366.19 | 3235.35 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 3422.50 | 3366.19 | 3235.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 10:15:00 | 3402.10 | 3370.62 | 3247.15 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-30 14:00:00 | 3400.00 | 3371.35 | 3249.96 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 3250.10 | 3367.26 | 3253.81 | SL hit (close<ema400) qty=1.00 sl=3253.81 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-08-01 10:15:00 | 3248.90 | 3367.26 | 3253.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 3250.00 | 3366.09 | 3253.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 3252.10 | 3366.09 | 3253.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:00:00 | 3253.30 | 3363.83 | 3253.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 3218.70 | 3362.38 | 3253.59 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 13:15:00 | 3218.70 | 3362.38 | 3253.59 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 13:45:00 | 3251.60 | 3362.38 | 3253.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 3143.00 | 3360.20 | 3253.04 | SL hit (close<static) qty=1.00 sl=3240.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 2881.70 | 3173.78 | 3174.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 2852.10 | 3170.58 | 3172.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 3092.20 | 3089.02 | 3126.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 10:00:00 | 3092.20 | 3089.02 | 3126.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3036.70 | 3002.07 | 3058.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 3037.30 | 3002.07 | 3058.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 3061.30 | 3005.94 | 3056.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 3061.30 | 3005.94 | 3056.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 3050.40 | 3006.38 | 3056.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 3042.70 | 3006.81 | 3056.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 3042.40 | 3007.17 | 3056.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:45:00 | 3033.20 | 3008.73 | 3056.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 3041.70 | 3008.98 | 3056.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 3052.20 | 3010.06 | 3056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:15:00 | 3050.00 | 3010.06 | 3056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 3056.00 | 3010.52 | 3056.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:15:00 | 3058.00 | 3010.52 | 3056.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3058.00 | 3010.99 | 3056.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 3042.40 | 3010.99 | 3056.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 3038.00 | 3011.26 | 3056.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 3023.00 | 3011.26 | 3056.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 3028.50 | 3013.50 | 3055.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 10:15:00 | 3030.70 | 3013.50 | 3055.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 3030.90 | 3015.20 | 3054.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 3050.00 | 3016.26 | 3054.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 14:15:00 | 3029.70 | 3016.26 | 3054.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:00:00 | 3049.10 | 3016.59 | 3054.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3063.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3063.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3063.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3063.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3066.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3066.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3066.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3066.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3058.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 14:15:00 | 3078.40 | 3010.98 | 3043.49 | SL hit (close>static) qty=1.00 sl=3058.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 09:15:00 | 3044.80 | 3011.52 | 3043.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 3047.30 | 3012.84 | 3043.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 3070.30 | 3013.41 | 3043.60 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 3070.30 | 3013.41 | 3043.60 | SL hit (close>static) qty=1.00 sl=3058.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 3070.30 | 3013.41 | 3043.60 | SL hit (close>static) qty=1.00 sl=3058.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 3070.30 | 3013.41 | 3043.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 3025.00 | 3014.21 | 3043.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 11:15:00 | 3023.50 | 3014.21 | 3043.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 12:00:00 | 3021.00 | 3014.28 | 3043.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 09:30:00 | 3022.80 | 3014.48 | 3042.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 2997.50 | 3015.57 | 3042.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 3019.00 | 2979.88 | 3011.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:00:00 | 3019.00 | 2979.88 | 3011.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 3020.20 | 2980.28 | 3011.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:30:00 | 3020.00 | 2980.28 | 3011.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 3019.00 | 2986.77 | 3013.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:45:00 | 3022.50 | 2986.77 | 3013.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 3018.60 | 2987.09 | 3013.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 3018.60 | 2987.09 | 3013.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 3015.00 | 2987.60 | 3013.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 3030.70 | 2987.60 | 3013.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3015.00 | 2987.87 | 3013.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:30:00 | 3005.00 | 2988.26 | 3013.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 3055.00 | 2988.93 | 3013.34 | SL hit (close>static) qty=1.00 sl=3049.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 3055.00 | 2988.93 | 3013.34 | SL hit (close>static) qty=1.00 sl=3049.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 3055.00 | 2988.93 | 3013.34 | SL hit (close>static) qty=1.00 sl=3049.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 3055.00 | 2988.93 | 3013.34 | SL hit (close>static) qty=1.00 sl=3049.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-30 11:15:00 | 3055.00 | 2988.93 | 3013.34 | SL hit (close>static) qty=1.00 sl=3036.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 2983.20 | 3005.60 | 3019.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 3042.50 | 3006.15 | 3019.99 | SL hit (close>static) qty=1.00 sl=3036.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:30:00 | 2996.40 | 3008.11 | 3020.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 3041.00 | 3008.63 | 3020.78 | SL hit (close>static) qty=1.00 sl=3036.40 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 3005.80 | 3010.80 | 3021.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 3023.20 | 3010.68 | 3021.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 3027.60 | 3010.68 | 3021.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 3047.60 | 3011.04 | 3021.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-07 13:15:00 | 3047.60 | 3011.04 | 3021.48 | SL hit (close>static) qty=1.00 sl=3036.40 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 3044.50 | 3011.04 | 3021.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3016.00 | 3011.09 | 3021.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 3046.30 | 3011.09 | 3021.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3005.00 | 3011.03 | 3021.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3034.30 | 3011.03 | 3021.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 3016.10 | 3011.08 | 3021.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 11:00:00 | 3009.60 | 3011.07 | 3021.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 3057.90 | 3013.53 | 3021.95 | SL hit (close>static) qty=1.00 sl=3054.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 3100.00 | 3029.03 | 3028.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 3153.40 | 3041.41 | 3035.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 3069.50 | 3079.19 | 3059.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 3069.50 | 3079.19 | 3059.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 3035.00 | 3078.75 | 3059.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 3035.00 | 3078.75 | 3059.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 3024.10 | 3078.20 | 3059.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 3023.40 | 3078.20 | 3059.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 3012.50 | 3073.32 | 3057.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 3031.40 | 3073.32 | 3057.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 3050.00 | 3072.92 | 3057.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:45:00 | 3050.00 | 3072.92 | 3057.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 3050.50 | 3066.48 | 3055.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 3050.00 | 3066.48 | 3055.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 3059.80 | 3066.08 | 3055.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 3080.80 | 3066.08 | 3055.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 3083.60 | 3066.26 | 3055.41 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 2977.30 | 3050.30 | 3050.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 14:15:00 | 2948.00 | 3048.72 | 3049.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 3050.20 | 3042.74 | 3046.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 3050.20 | 3042.74 | 3046.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 3047.60 | 3042.79 | 3046.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 3054.70 | 3042.79 | 3046.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 3050.00 | 3042.86 | 3046.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 3032.90 | 3043.10 | 3046.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 3060.30 | 3043.27 | 3046.50 | SL hit (close>static) qty=1.00 sl=3052.20 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 12:15:00 | 3030.60 | 3047.69 | 3048.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 3042.40 | 3046.85 | 3048.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 3033.60 | 3030.07 | 3038.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 3049.30 | 3030.27 | 3038.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 3055.60 | 3030.27 | 3038.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 3050.00 | 3030.46 | 3039.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 3050.00 | 3030.46 | 3039.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 3052.30 | 3030.68 | 3039.10 | SL hit (close>static) qty=1.00 sl=3052.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 3052.30 | 3030.68 | 3039.10 | SL hit (close>static) qty=1.00 sl=3052.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-14 12:15:00 | 3052.30 | 3030.68 | 3039.10 | SL hit (close>static) qty=1.00 sl=3052.20 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 3029.80 | 3021.08 | 3032.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 14:00:00 | 3029.80 | 3021.08 | 3032.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 3056.10 | 3021.43 | 3033.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 3056.10 | 3021.43 | 3033.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 3050.00 | 3021.71 | 3033.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 3075.00 | 3021.71 | 3033.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 3028.80 | 3021.86 | 3033.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 3032.10 | 3021.86 | 3033.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 3020.30 | 3021.92 | 3032.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 3008.80 | 3021.92 | 3032.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 3004.00 | 3021.79 | 3032.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 3012.60 | 3021.79 | 3032.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 14:00:00 | 3013.00 | 3021.28 | 3032.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 3045.00 | 3021.54 | 3032.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 3045.00 | 3021.54 | 3032.23 | SL hit (close>static) qty=1.00 sl=3034.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 3045.00 | 3021.54 | 3032.23 | SL hit (close>static) qty=1.00 sl=3034.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 3045.00 | 3021.54 | 3032.23 | SL hit (close>static) qty=1.00 sl=3034.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-23 15:15:00 | 3045.00 | 3021.54 | 3032.23 | SL hit (close>static) qty=1.00 sl=3034.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 3015.40 | 3021.54 | 3032.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 3032.20 | 3021.65 | 3032.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:45:00 | 2971.00 | 3021.26 | 3031.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 2949.80 | 3020.89 | 3031.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 2962.00 | 3018.30 | 3030.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2966.50 | 3016.52 | 3028.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3040.50 | 3007.92 | 3023.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3281.80 | 3011.60 | 3024.40 | SL hit (close>static) qty=1.00 sl=3075.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3281.80 | 3011.60 | 3024.40 | SL hit (close>static) qty=1.00 sl=3075.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3281.80 | 3011.60 | 3024.40 | SL hit (close>static) qty=1.00 sl=3075.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3281.80 | 3011.60 | 3024.40 | SL hit (close>static) qty=1.00 sl=3075.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 13:15:00 | 3252.90 | 3037.02 | 3036.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 3293.50 | 3062.81 | 3050.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 3089.70 | 3098.62 | 3071.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 3086.30 | 3098.62 | 3071.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 3057.00 | 3098.35 | 3071.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 15:00:00 | 3057.00 | 3098.35 | 3071.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 3120.00 | 3098.56 | 3072.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-16 09:15:00 | 3038.90 | 3098.56 | 3072.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 3057.10 | 3098.15 | 3071.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 09:45:00 | 3087.70 | 3094.13 | 3070.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-27 09:15:00 | 3396.47 | 3155.08 | 3109.68 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 15:00:00 | 3107.50 | 3268.35 | 3204.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-01 10:15:00 | 3418.25 | 3262.00 | 3209.58 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-28 09:15:00 | 3422.50 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -5.04% |
| BUY | retest1 | 2025-07-30 10:15:00 | 3402.10 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest1 | 2025-07-30 14:00:00 | 3400.00 | 2025-08-01 09:15:00 | 3250.10 | STOP_HIT | 1.00 | -4.41% |
| BUY | retest2 | 2025-08-01 11:15:00 | 3252.10 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-01 13:00:00 | 3253.30 | 2025-08-01 13:15:00 | 3218.70 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-08-01 13:45:00 | 3251.60 | 2025-08-01 14:15:00 | 3143.00 | STOP_HIT | 1.00 | -3.34% |
| SELL | retest2 | 2025-09-16 11:30:00 | 3042.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-16 13:00:00 | 3042.40 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-17 09:45:00 | 3033.20 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-09-17 10:30:00 | 3041.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-18 10:15:00 | 3023.00 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-09-19 09:45:00 | 3028.50 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-09-19 10:15:00 | 3030.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-22 10:45:00 | 3030.90 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-09-22 14:15:00 | 3029.70 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-22 15:00:00 | 3049.10 | 2025-10-01 14:15:00 | 3078.40 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-03 09:15:00 | 3044.80 | 2025-10-03 14:15:00 | 3070.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-10-03 14:15:00 | 3047.30 | 2025-10-03 14:15:00 | 3070.30 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-10-06 11:15:00 | 3023.50 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-06 12:00:00 | 3021.00 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-07 09:30:00 | 3022.80 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-08 09:15:00 | 2997.50 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-10-30 10:30:00 | 3005.00 | 2025-10-30 11:15:00 | 3055.00 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2025-11-04 09:30:00 | 2983.20 | 2025-11-04 11:15:00 | 3042.50 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-11-06 09:30:00 | 2996.40 | 2025-11-06 11:15:00 | 3041.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-11-07 09:15:00 | 3005.80 | 2025-11-07 13:15:00 | 3047.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-11-10 11:00:00 | 3009.60 | 2025-11-11 15:15:00 | 3057.90 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-01-05 09:15:00 | 3032.90 | 2026-01-05 09:15:00 | 3060.30 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-01-07 12:15:00 | 3030.60 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-08 10:30:00 | 3042.40 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2026-01-14 10:00:00 | 3033.60 | 2026-01-14 12:15:00 | 3052.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-01-22 13:15:00 | 3008.80 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-01-23 09:30:00 | 3004.00 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-23 10:15:00 | 3012.60 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-01-23 14:00:00 | 3013.00 | 2026-01-23 15:15:00 | 3045.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-27 14:45:00 | 2971.00 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.46% |
| SELL | retest2 | 2026-01-28 09:15:00 | 2949.80 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -11.26% |
| SELL | retest2 | 2026-01-28 13:15:00 | 2962.00 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.80% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2966.50 | 2026-02-03 09:15:00 | 3281.80 | STOP_HIT | 1.00 | -10.63% |
| BUY | retest2 | 2026-02-17 09:45:00 | 3087.70 | 2026-02-27 09:15:00 | 3396.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 15:00:00 | 3107.50 | 2026-04-01 10:15:00 | 3418.25 | TARGET_HIT | 1.00 | 10.00% |
