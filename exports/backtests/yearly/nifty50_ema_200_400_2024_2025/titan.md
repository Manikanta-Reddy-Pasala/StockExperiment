# TITAN (TITAN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4517.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 11 |
| ALERT2_SKIP | 2 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 50 |
| PARTIAL | 5 |
| TARGET_HIT | 2 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 43
- **Target hits / Stop hits / Partials:** 2 / 48 / 5
- **Avg / median % per leg:** -0.51% / -1.30%
- **Sum % (uncompounded):** -28.01%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.29% | -5.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 2 | 11.1% | 2 | 16 | 0 | -0.29% | -5.2% |
| SELL (all) | 37 | 10 | 27.0% | 0 | 32 | 5 | -0.62% | -22.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 10 | 27.0% | 0 | 32 | 5 | -0.62% | -22.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 55 | 12 | 21.8% | 2 | 48 | 5 | -0.51% | -28.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 11:15:00 | 3613.00 | 3398.93 | 3398.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 13:15:00 | 3631.90 | 3427.21 | 3413.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 12:15:00 | 3682.05 | 3695.85 | 3604.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 13:00:00 | 3682.05 | 3695.85 | 3604.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 3582.00 | 3694.61 | 3609.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 3582.00 | 3694.61 | 3609.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 3584.45 | 3693.51 | 3608.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:00:00 | 3584.45 | 3693.51 | 3608.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 3592.50 | 3692.51 | 3608.79 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-10-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 14:15:00 | 3329.25 | 3555.66 | 3556.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 09:15:00 | 3309.80 | 3550.99 | 3554.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 14:15:00 | 3313.80 | 3297.77 | 3386.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 14:45:00 | 3307.50 | 3297.77 | 3386.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 3368.10 | 3295.18 | 3364.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 3362.95 | 3295.18 | 3364.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 3367.25 | 3295.90 | 3364.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 3348.05 | 3295.90 | 3364.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 12:15:00 | 3356.05 | 3296.49 | 3364.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 3353.45 | 3371.69 | 3387.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 3386.40 | 3371.70 | 3387.77 | SL hit (close>static) qty=1.00 sl=3375.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 13:15:00 | 3443.75 | 3388.82 | 3388.75 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 13:15:00 | 3321.50 | 3388.49 | 3388.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 14:15:00 | 3316.80 | 3387.78 | 3388.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3381.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3381.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 3394.00 | 3374.61 | 3381.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 3394.00 | 3374.61 | 3381.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 3392.95 | 3374.79 | 3381.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 12:30:00 | 3387.15 | 3375.14 | 3381.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 13:45:00 | 3385.45 | 3375.33 | 3381.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:45:00 | 3387.30 | 3375.38 | 3381.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:30:00 | 3372.00 | 3375.54 | 3381.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 3381.05 | 3375.33 | 3381.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:00:00 | 3381.05 | 3375.33 | 3381.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 3387.55 | 3375.45 | 3381.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:00:00 | 3387.55 | 3375.45 | 3381.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 3387.70 | 3375.58 | 3381.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 13:30:00 | 3398.25 | 3375.58 | 3381.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 3354.30 | 3375.36 | 3381.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 13:15:00 | 3339.65 | 3374.78 | 3380.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 3392.60 | 3374.22 | 3380.33 | SL hit (close>static) qty=1.00 sl=3388.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 3575.60 | 3384.84 | 3384.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 3623.15 | 3392.76 | 3388.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 3408.15 | 3420.62 | 3403.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 3408.15 | 3420.62 | 3403.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 3406.10 | 3420.47 | 3403.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:30:00 | 3405.10 | 3420.47 | 3403.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 3438.10 | 3420.57 | 3404.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 3447.45 | 3420.78 | 3404.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 3448.10 | 3421.00 | 3404.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 3377.75 | 3420.91 | 3404.92 | SL hit (close<static) qty=1.00 sl=3388.20 alert=retest2 |

### Cycle 6 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 3229.85 | 3390.32 | 3390.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 3207.70 | 3385.38 | 3388.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 3167.65 | 3154.29 | 3235.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:00:00 | 3167.65 | 3154.29 | 3235.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3133.45 | 3110.42 | 3182.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 3108.65 | 3110.64 | 3181.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 3121.80 | 3110.64 | 3181.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 15:00:00 | 3120.60 | 3111.76 | 3181.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 11:15:00 | 3228.75 | 3118.61 | 3180.88 | SL hit (close>static) qty=1.00 sl=3227.25 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 3379.70 | 3223.88 | 3223.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 11:15:00 | 3402.60 | 3237.64 | 3230.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 3499.00 | 3499.42 | 3425.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 3499.00 | 3499.42 | 3425.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3408.90 | 3497.31 | 3426.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 3408.00 | 3497.31 | 3426.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 3423.80 | 3496.58 | 3426.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:30:00 | 3438.40 | 3494.58 | 3426.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 3438.40 | 3490.49 | 3426.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 3442.90 | 3490.01 | 3426.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:45:00 | 3442.80 | 3489.48 | 3426.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3409.40 | 3488.20 | 3426.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:30:00 | 3448.00 | 3483.02 | 3426.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:45:00 | 3464.80 | 3480.68 | 3426.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 3400.10 | 3550.73 | 3496.65 | SL hit (close<static) qty=1.00 sl=3402.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 3355.10 | 3464.89 | 3465.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 3328.50 | 3460.14 | 3462.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 11:15:00 | 3449.00 | 3441.68 | 3452.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 12:00:00 | 3449.00 | 3441.68 | 3452.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 3453.00 | 3441.79 | 3452.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:30:00 | 3438.60 | 3441.67 | 3452.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 3469.40 | 3439.12 | 3450.49 | SL hit (close>static) qty=1.00 sl=3462.40 alert=retest2 |

### Cycle 9 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 3559.00 | 3459.85 | 3459.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 3569.70 | 3462.87 | 3460.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 3576.80 | 3579.54 | 3536.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 12:45:00 | 3577.50 | 3579.54 | 3536.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3526.10 | 3578.50 | 3537.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 3526.10 | 3578.50 | 3537.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 3535.30 | 3578.07 | 3537.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 3526.50 | 3578.07 | 3537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 3536.60 | 3577.66 | 3537.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 12:45:00 | 3530.80 | 3577.66 | 3537.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 3540.70 | 3577.29 | 3537.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:30:00 | 3545.00 | 3576.85 | 3537.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 15:15:00 | 3532.00 | 3576.41 | 3537.05 | SL hit (close<static) qty=1.00 sl=3532.40 alert=retest2 |

### Cycle 10 — SELL (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 11:15:00 | 3380.00 | 3511.17 | 3511.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 09:15:00 | 3361.00 | 3496.94 | 3504.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3571.00 | 3479.34 | 3493.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 3571.00 | 3479.34 | 3493.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3560.00 | 3480.14 | 3494.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 12:15:00 | 3539.00 | 3480.94 | 3494.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 3547.90 | 3484.33 | 3495.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 3555.40 | 3489.09 | 3497.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 12:45:00 | 3553.50 | 3491.24 | 3498.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 3636.30 | 3501.02 | 3503.16 | SL hit (close>static) qty=1.00 sl=3578.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 3641.40 | 3506.57 | 3505.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 3694.00 | 3511.02 | 3508.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 13:15:00 | 3810.00 | 3813.27 | 3724.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 14:00:00 | 3810.00 | 3813.27 | 3724.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 3999.00 | 4056.22 | 3958.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 09:15:00 | 4011.70 | 4055.69 | 3958.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 09:15:00 | 3891.70 | 4048.75 | 3958.75 | SL hit (close<static) qty=1.00 sl=3955.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-20 10:15:00 | 3432.05 | 2024-07-05 10:15:00 | 3267.95 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2024-06-20 13:45:00 | 3429.35 | 2024-07-05 10:15:00 | 3267.57 | PARTIAL | 0.50 | 4.72% |
| SELL | retest2 | 2024-06-20 14:30:00 | 3439.95 | 2024-07-05 14:15:00 | 3260.45 | PARTIAL | 0.50 | 5.22% |
| SELL | retest2 | 2024-06-21 09:30:00 | 3439.55 | 2024-07-05 14:15:00 | 3257.88 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2024-07-02 11:45:00 | 3404.95 | 2024-07-08 09:15:00 | 3234.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-20 10:15:00 | 3432.05 | 2024-07-23 11:15:00 | 3378.90 | STOP_HIT | 0.50 | 1.55% |
| SELL | retest2 | 2024-06-20 13:45:00 | 3429.35 | 2024-07-23 11:15:00 | 3378.90 | STOP_HIT | 0.50 | 1.47% |
| SELL | retest2 | 2024-06-20 14:30:00 | 3439.95 | 2024-07-23 11:15:00 | 3378.90 | STOP_HIT | 0.50 | 1.77% |
| SELL | retest2 | 2024-06-21 09:30:00 | 3439.55 | 2024-07-23 11:15:00 | 3378.90 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2024-07-02 11:45:00 | 3404.95 | 2024-07-23 11:15:00 | 3378.90 | STOP_HIT | 0.50 | 0.77% |
| SELL | retest2 | 2024-07-25 10:00:00 | 3407.00 | 2024-07-26 10:15:00 | 3456.45 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-07-25 11:45:00 | 3404.20 | 2024-07-26 10:15:00 | 3456.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-07-26 09:30:00 | 3404.20 | 2024-07-26 10:15:00 | 3456.45 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2024-08-05 13:45:00 | 3369.00 | 2024-08-16 11:15:00 | 3413.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-06 13:30:00 | 3363.65 | 2024-08-16 11:15:00 | 3413.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2024-12-19 15:00:00 | 3353.45 | 2024-12-20 09:15:00 | 3386.40 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-20 14:00:00 | 3354.70 | 2024-12-23 10:15:00 | 3397.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-12-20 15:00:00 | 3352.30 | 2024-12-23 10:15:00 | 3397.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-12-23 09:45:00 | 3341.25 | 2024-12-23 10:15:00 | 3397.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-12-24 12:00:00 | 3381.00 | 2025-01-03 09:15:00 | 3452.65 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-01-20 12:30:00 | 3387.15 | 2025-01-23 10:15:00 | 3392.60 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-01-20 13:45:00 | 3385.45 | 2025-01-23 11:15:00 | 3409.10 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-01-20 14:45:00 | 3387.30 | 2025-01-23 11:15:00 | 3409.10 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-01-21 09:30:00 | 3372.00 | 2025-01-23 11:15:00 | 3409.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-01-22 13:15:00 | 3339.65 | 2025-01-23 11:15:00 | 3409.10 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-01-27 13:15:00 | 3342.10 | 2025-01-31 09:15:00 | 3485.40 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2025-01-28 12:30:00 | 3342.30 | 2025-01-31 09:15:00 | 3485.40 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-01-28 13:15:00 | 3336.00 | 2025-01-31 09:15:00 | 3485.40 | STOP_HIT | 1.00 | -4.48% |
| BUY | retest2 | 2025-02-07 10:45:00 | 3447.45 | 2025-02-10 09:15:00 | 3377.75 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-02-07 11:30:00 | 3448.10 | 2025-02-10 09:15:00 | 3377.75 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-08 10:30:00 | 3108.65 | 2025-04-11 11:15:00 | 3228.75 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-04-08 11:15:00 | 3121.80 | 2025-04-11 11:15:00 | 3228.75 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2025-04-08 15:00:00 | 3120.60 | 2025-04-11 11:15:00 | 3228.75 | STOP_HIT | 1.00 | -3.47% |
| BUY | retest2 | 2025-06-13 13:30:00 | 3438.40 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-16 13:15:00 | 3438.40 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-06-16 14:00:00 | 3442.90 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-16 14:45:00 | 3442.80 | 2025-07-11 09:15:00 | 3400.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-06-18 09:30:00 | 3448.00 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-06-18 14:45:00 | 3464.80 | 2025-07-11 10:15:00 | 3390.70 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-06 13:30:00 | 3438.60 | 2025-08-08 09:15:00 | 3469.40 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-11 09:15:00 | 3420.60 | 2025-08-11 11:15:00 | 3465.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-15 14:30:00 | 3545.00 | 2025-09-15 15:15:00 | 3532.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-09-16 12:45:00 | 3544.20 | 2025-09-17 10:15:00 | 3527.00 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-09-16 14:00:00 | 3557.00 | 2025-09-17 10:15:00 | 3527.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-09-18 10:00:00 | 3544.70 | 2025-09-18 11:15:00 | 3528.50 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-08 12:15:00 | 3539.00 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-10-09 09:15:00 | 3547.90 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-10-10 10:15:00 | 3555.40 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-10-10 12:45:00 | 3553.50 | 2025-10-16 09:15:00 | 3636.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-28 09:15:00 | 4011.70 | 2026-01-29 09:15:00 | 3891.70 | STOP_HIT | 1.00 | -2.99% |
| BUY | retest2 | 2026-02-01 13:00:00 | 4044.90 | 2026-02-01 15:15:00 | 3944.00 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-02-03 09:15:00 | 4070.00 | 2026-03-23 09:15:00 | 3946.40 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-25 10:00:00 | 4013.00 | 2026-03-30 10:15:00 | 3936.70 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2026-04-02 15:00:00 | 4100.40 | 2026-04-10 12:15:00 | 4510.44 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 09:15:00 | 4152.60 | 2026-05-08 14:15:00 | 4567.86 | TARGET_HIT | 1.00 | 10.00% |
