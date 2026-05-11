# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3838.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 16 |
| ALERT2 | 15 |
| ALERT2_SKIP | 6 |
| ALERT3 | 66 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 54 |
| PARTIAL | 4 |
| TARGET_HIT | 5 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 47
- **Target hits / Stop hits / Partials:** 5 / 49 / 4
- **Avg / median % per leg:** -0.27% / -1.59%
- **Sum % (uncompounded):** -15.59%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 4 | 12.9% | 4 | 27 | 0 | -0.43% | -13.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 4 | 12.9% | 4 | 27 | 0 | -0.43% | -13.4% |
| SELL (all) | 27 | 7 | 25.9% | 1 | 22 | 4 | -0.08% | -2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 1 | 22 | 4 | -0.08% | -2.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 11 | 19.0% | 5 | 49 | 4 | -0.27% | -15.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-05 11:15:00 | 1758.43 | 1857.38 | 1857.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 13:15:00 | 1751.49 | 1855.34 | 1856.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 11:15:00 | 1747.12 | 1725.29 | 1766.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 11:15:00 | 1747.12 | 1725.29 | 1766.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 11:15:00 | 1747.12 | 1725.29 | 1766.79 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 12:15:00 | 1913.19 | 1785.35 | 1785.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 10:15:00 | 1935.39 | 1821.53 | 1804.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3273.66 | 3298.57 | 3013.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 3273.66 | 3298.57 | 3013.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 3203.54 | 3295.41 | 3014.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 13:15:00 | 3245.02 | 3295.41 | 3014.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 11:15:00 | 3221.19 | 3289.64 | 3018.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 13:45:00 | 3217.48 | 3286.89 | 3021.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 10:15:00 | 3543.31 | 3320.15 | 3078.41 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 3283.60 | 3451.77 | 3452.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 3270.78 | 3448.41 | 3450.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 3393.02 | 3388.78 | 3414.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 11:00:00 | 3393.02 | 3388.78 | 3414.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 3452.63 | 3389.63 | 3414.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 3452.63 | 3389.63 | 3414.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 3442.69 | 3390.16 | 3414.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 11:30:00 | 3433.24 | 3390.69 | 3414.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 13:15:00 | 3426.28 | 3391.18 | 3414.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 15:15:00 | 3458.10 | 3392.66 | 3415.04 | SL hit (close>static) qty=1.00 sl=3455.14 alert=retest2 |

### Cycle 4 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 3616.44 | 3435.06 | 3434.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 12:15:00 | 3634.09 | 3437.04 | 3435.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 14:15:00 | 3477.96 | 3487.51 | 3463.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-07 14:30:00 | 3477.56 | 3487.51 | 3463.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 3532.15 | 3487.87 | 3463.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 10:15:00 | 3546.94 | 3487.87 | 3463.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-15 09:15:00 | 3901.63 | 3579.55 | 3516.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 3327.62 | 3518.93 | 3519.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 10:15:00 | 3314.08 | 3515.14 | 3517.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 3571.94 | 3468.77 | 3492.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 3571.94 | 3468.77 | 3492.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 3571.94 | 3468.77 | 3492.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 3590.83 | 3468.77 | 3492.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 3620.36 | 3470.28 | 3492.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:00:00 | 3620.36 | 3470.28 | 3492.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 13:15:00 | 3699.04 | 3512.98 | 3512.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 3723.57 | 3518.52 | 3515.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 11:15:00 | 3710.64 | 3747.76 | 3661.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 12:00:00 | 3710.64 | 3747.76 | 3661.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 3526.75 | 3745.56 | 3660.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:30:00 | 3480.64 | 3745.56 | 3660.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 3459.61 | 3742.71 | 3659.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 3486.98 | 3742.71 | 3659.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 13:15:00 | 3266.67 | 3590.12 | 3591.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 3250.92 | 3586.74 | 3589.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 12:15:00 | 2562.25 | 2555.22 | 2755.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:00:00 | 2562.25 | 2555.22 | 2755.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2694.14 | 2565.82 | 2730.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:30:00 | 2710.60 | 2565.82 | 2730.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 12:15:00 | 2736.43 | 2569.04 | 2730.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 12:30:00 | 2735.38 | 2569.04 | 2730.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 2709.41 | 2570.44 | 2730.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2690.52 | 2570.44 | 2730.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:45:00 | 2692.01 | 2571.72 | 2730.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 2557.41 | 2581.75 | 2718.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 10:15:00 | 2602.52 | 2581.95 | 2718.19 | SL hit (close>ema200) qty=0.50 sl=2581.95 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 2918.50 | 2767.91 | 2767.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 12:15:00 | 2950.90 | 2773.08 | 2770.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 3208.60 | 3220.48 | 3092.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:00:00 | 3208.60 | 3220.48 | 3092.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3097.90 | 3219.03 | 3096.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 3097.90 | 3219.03 | 3096.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3117.50 | 3218.02 | 3096.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 3142.20 | 3212.45 | 3096.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 14:30:00 | 3144.90 | 3205.39 | 3100.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:30:00 | 3151.00 | 3203.74 | 3100.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:00:00 | 3153.40 | 3202.69 | 3101.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 3147.00 | 3234.15 | 3151.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 3147.00 | 3234.15 | 3151.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 3119.00 | 3233.00 | 3151.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:30:00 | 3110.90 | 3233.00 | 3151.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3120.00 | 3225.67 | 3150.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 11:00:00 | 3120.00 | 3225.67 | 3150.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-14 11:15:00 | 3077.40 | 3224.19 | 3150.33 | SL hit (close<static) qty=1.00 sl=3086.20 alert=retest2 |

### Cycle 9 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 3100.50 | 3117.64 | 3117.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 3085.40 | 3117.32 | 3117.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3118.50 | 3114.59 | 3116.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 3117.10 | 3114.59 | 3116.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3120.90 | 3114.65 | 3116.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3111.00 | 3114.65 | 3116.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3125.00 | 3114.76 | 3116.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:15:00 | 3101.60 | 3114.66 | 3116.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-11 09:15:00 | 2946.52 | 3111.59 | 3114.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 3124.00 | 3111.72 | 3114.55 | SL hit (close>ema200) qty=0.50 sl=3111.72 alert=retest2 |

### Cycle 10 — BUY (started 2025-08-14 13:15:00)

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
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 09:30:00 | 3153.00 | 3122.82 | 3121.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 11:15:00 | 3106.20 | 3135.33 | 3128.09 | SL hit (close<static) qty=1.00 sl=3116.10 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-28 11:15:00)

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

### Cycle 12 — BUY (started 2025-11-26 10:15:00)

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

### Cycle 13 — SELL (started 2025-12-26 10:15:00)

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

### Cycle 14 — BUY (started 2026-02-17 13:15:00)

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

### Cycle 15 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 2963.70 | 3151.96 | 3152.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2937.50 | 3149.83 | 3151.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.66 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 3360.20 | 3149.48 | 3148.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 3504.10 | 3161.01 | 3154.48 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-04 13:15:00 | 3245.02 | 2024-06-12 10:15:00 | 3543.31 | TARGET_HIT | 1.00 | 9.19% |
| BUY | retest2 | 2024-06-05 11:15:00 | 3221.19 | 2024-06-12 10:15:00 | 3539.23 | TARGET_HIT | 1.00 | 9.87% |
| BUY | retest2 | 2024-06-05 13:45:00 | 3217.48 | 2024-06-13 09:15:00 | 3569.52 | TARGET_HIT | 1.00 | 10.94% |
| SELL | retest2 | 2024-09-23 11:30:00 | 3433.24 | 2024-09-23 15:15:00 | 3458.10 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-09-23 13:15:00 | 3426.28 | 2024-09-23 15:15:00 | 3458.10 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-10-08 10:15:00 | 3546.94 | 2024-10-15 09:15:00 | 3901.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-08 09:30:00 | 3549.65 | 2024-11-12 09:15:00 | 3441.62 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-11-08 11:00:00 | 3533.79 | 2024-11-12 09:15:00 | 3441.62 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2024-11-11 10:15:00 | 3546.32 | 2024-11-12 09:15:00 | 3441.62 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2025-03-26 14:15:00 | 2690.52 | 2025-04-02 09:15:00 | 2557.41 | PARTIAL | 0.50 | 4.95% |
| SELL | retest2 | 2025-03-26 14:15:00 | 2690.52 | 2025-04-02 10:15:00 | 2602.52 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-03-26 14:45:00 | 2692.01 | 2025-04-04 09:15:00 | 2555.99 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2025-03-26 14:45:00 | 2692.01 | 2025-04-04 14:15:00 | 2421.47 | TARGET_HIT | 0.50 | 10.05% |
| SELL | retest2 | 2025-04-09 09:45:00 | 2701.00 | 2025-04-11 10:15:00 | 2769.70 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-04-09 11:45:00 | 2701.70 | 2025-04-11 10:15:00 | 2769.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-11 13:45:00 | 2724.35 | 2025-04-15 11:15:00 | 2823.00 | STOP_HIT | 1.00 | -3.62% |
| SELL | retest2 | 2025-04-11 15:15:00 | 2726.00 | 2025-04-15 11:15:00 | 2823.00 | STOP_HIT | 1.00 | -3.56% |
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
