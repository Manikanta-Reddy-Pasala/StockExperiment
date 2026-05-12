# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 5560.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 215 |
| ALERT1 | 140 |
| ALERT2 | 138 |
| ALERT2_SKIP | 56 |
| ALERT3 | 383 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 197 |
| PARTIAL | 9 |
| TARGET_HIT | 2 |
| STOP_HIT | 201 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 212 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 53 / 159
- **Target hits / Stop hits / Partials:** 2 / 201 / 9
- **Avg / median % per leg:** -0.11% / -0.77%
- **Sum % (uncompounded):** -23.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 111 | 29 | 26.1% | 2 | 109 | 0 | -0.07% | -7.5% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.00% | -4.0% |
| BUY @ 3rd Alert (retest2) | 107 | 29 | 27.1% | 2 | 105 | 0 | -0.03% | -3.5% |
| SELL (all) | 101 | 24 | 23.8% | 0 | 92 | 9 | -0.16% | -16.2% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.07% | -0.1% |
| SELL @ 3rd Alert (retest2) | 99 | 23 | 23.2% | 0 | 90 | 9 | -0.16% | -16.1% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.69% | -4.1% |
| retest2 (combined) | 206 | 52 | 25.2% | 2 | 195 | 9 | -0.09% | -19.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 3300.50 | 3290.46 | 3289.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 15:15:00 | 3340.30 | 3314.22 | 3304.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-29 13:15:00 | 3350.90 | 3354.38 | 3339.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-29 14:00:00 | 3350.90 | 3354.38 | 3339.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 09:15:00 | 3377.75 | 3358.80 | 3344.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 10:30:00 | 3381.30 | 3363.04 | 3348.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-30 11:00:00 | 3380.00 | 3363.04 | 3348.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:15:00 | 3388.00 | 3363.66 | 3354.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-31 10:45:00 | 3381.70 | 3367.83 | 3357.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 12:15:00 | 3356.05 | 3365.08 | 3357.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 13:00:00 | 3356.05 | 3365.08 | 3357.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 13:15:00 | 3351.30 | 3362.32 | 3357.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 14:00:00 | 3351.30 | 3362.32 | 3357.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 3354.10 | 3360.68 | 3357.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 3354.10 | 3360.68 | 3357.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 3368.00 | 3362.14 | 3358.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 09:15:00 | 3410.15 | 3362.14 | 3358.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 14:30:00 | 3379.05 | 3375.33 | 3368.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-02 11:00:00 | 3378.95 | 3378.37 | 3371.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 12:15:00 | 3342.15 | 3367.10 | 3367.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-02 12:15:00 | 3342.15 | 3367.10 | 3367.30 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 10:15:00 | 3375.75 | 3368.32 | 3367.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-05 14:15:00 | 3399.75 | 3375.41 | 3371.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 11:15:00 | 3374.65 | 3381.94 | 3376.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-06 11:15:00 | 3374.65 | 3381.94 | 3376.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 3374.65 | 3381.94 | 3376.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:45:00 | 3375.95 | 3381.94 | 3376.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 12:15:00 | 3379.75 | 3381.50 | 3376.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 13:30:00 | 3385.60 | 3381.81 | 3377.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 09:15:00 | 3385.45 | 3411.92 | 3412.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 09:15:00 | 3385.45 | 3411.92 | 3412.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 10:15:00 | 3369.80 | 3403.50 | 3408.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 09:15:00 | 3423.95 | 3385.81 | 3394.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 09:15:00 | 3423.95 | 3385.81 | 3394.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 3423.95 | 3385.81 | 3394.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:00:00 | 3423.95 | 3385.81 | 3394.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 3438.00 | 3396.25 | 3398.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 11:00:00 | 3438.00 | 3396.25 | 3398.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 11:15:00 | 3440.00 | 3405.00 | 3402.45 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 15:15:00 | 3382.10 | 3400.82 | 3401.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-14 11:15:00 | 3380.00 | 3396.41 | 3399.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 10:15:00 | 3402.85 | 3385.91 | 3391.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 10:15:00 | 3402.85 | 3385.91 | 3391.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 10:15:00 | 3402.85 | 3385.91 | 3391.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 11:00:00 | 3402.85 | 3385.91 | 3391.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 11:15:00 | 3412.00 | 3391.13 | 3393.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:00:00 | 3412.00 | 3391.13 | 3393.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2023-06-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 12:15:00 | 3417.30 | 3396.36 | 3395.23 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 3392.00 | 3405.94 | 3406.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 10:15:00 | 3370.00 | 3382.42 | 3388.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 12:15:00 | 3352.00 | 3351.29 | 3365.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-23 13:00:00 | 3352.00 | 3351.29 | 3365.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 13:15:00 | 3357.85 | 3352.60 | 3364.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:00:00 | 3357.85 | 3352.60 | 3364.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 14:15:00 | 3355.00 | 3353.08 | 3363.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-23 14:45:00 | 3359.75 | 3353.08 | 3363.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 09:15:00 | 3369.95 | 3355.96 | 3363.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 09:45:00 | 3365.05 | 3355.96 | 3363.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 10:15:00 | 3391.05 | 3362.98 | 3365.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-26 10:30:00 | 3400.05 | 3362.98 | 3365.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 3390.65 | 3368.51 | 3367.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 14:15:00 | 3407.95 | 3381.46 | 3374.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 14:15:00 | 3393.00 | 3394.94 | 3386.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 15:00:00 | 3393.00 | 3394.94 | 3386.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 11:15:00 | 3461.30 | 3480.31 | 3459.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 11:45:00 | 3453.25 | 3480.31 | 3459.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 12:15:00 | 3461.90 | 3476.63 | 3460.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-03 12:45:00 | 3460.35 | 3476.63 | 3460.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 13:15:00 | 3495.85 | 3480.47 | 3463.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 09:15:00 | 3509.55 | 3485.94 | 3468.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 11:45:00 | 3504.45 | 3495.70 | 3478.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 13:30:00 | 3507.10 | 3497.56 | 3482.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-04 15:15:00 | 3504.00 | 3497.24 | 3483.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 3504.00 | 3498.59 | 3485.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-05 09:15:00 | 3495.00 | 3498.59 | 3485.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 3505.20 | 3499.91 | 3487.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 12:15:00 | 3510.95 | 3500.76 | 3489.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-05 14:45:00 | 3517.95 | 3505.07 | 3494.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-07 14:15:00 | 3512.80 | 3513.60 | 3511.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 14:15:00 | 3493.95 | 3509.67 | 3510.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 14:15:00 | 3493.95 | 3509.67 | 3510.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 14:15:00 | 3485.05 | 3497.12 | 3502.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 15:15:00 | 3500.00 | 3497.70 | 3502.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:15:00 | 3505.95 | 3497.70 | 3502.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 3515.35 | 3501.23 | 3503.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 3519.75 | 3501.23 | 3503.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 3504.25 | 3501.83 | 3503.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 3512.60 | 3501.83 | 3503.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 3515.40 | 3504.55 | 3504.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 12:00:00 | 3515.40 | 3504.55 | 3504.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 3511.25 | 3505.89 | 3505.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 12:15:00 | 3521.95 | 3514.22 | 3510.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 3504.05 | 3525.90 | 3520.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 13:15:00 | 3504.05 | 3525.90 | 3520.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 13:15:00 | 3504.05 | 3525.90 | 3520.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-13 14:00:00 | 3504.05 | 3525.90 | 3520.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 14:15:00 | 3512.65 | 3523.25 | 3519.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-14 09:30:00 | 3523.55 | 3518.44 | 3518.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-14 10:15:00 | 3508.85 | 3516.52 | 3517.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 10:15:00 | 3508.85 | 3516.52 | 3517.27 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 13:15:00 | 3522.75 | 3518.63 | 3518.08 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-14 15:15:00 | 3513.45 | 3517.71 | 3517.76 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 3567.55 | 3527.68 | 3522.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 3614.50 | 3548.74 | 3533.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-19 10:15:00 | 3634.80 | 3644.03 | 3614.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-19 11:00:00 | 3634.80 | 3644.03 | 3614.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 3617.75 | 3638.77 | 3614.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:45:00 | 3614.95 | 3638.77 | 3614.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 3628.05 | 3636.63 | 3615.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:00:00 | 3628.05 | 3636.63 | 3615.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 3670.20 | 3688.00 | 3662.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 09:15:00 | 3684.15 | 3688.00 | 3662.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 3681.35 | 3686.67 | 3664.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 10:45:00 | 3695.00 | 3688.04 | 3666.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 12:15:00 | 3693.95 | 3688.22 | 3668.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 09:15:00 | 3946.00 | 3974.14 | 3977.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2023-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 09:15:00 | 3946.00 | 3974.14 | 3977.12 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 4072.00 | 3969.28 | 3967.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 13:15:00 | 4083.95 | 4043.60 | 4016.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 11:15:00 | 4100.00 | 4117.69 | 4087.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 11:15:00 | 4100.00 | 4117.69 | 4087.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 4100.00 | 4117.69 | 4087.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 12:00:00 | 4100.00 | 4117.69 | 4087.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 4158.00 | 4189.49 | 4167.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 13:45:00 | 4153.00 | 4189.49 | 4167.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 14:15:00 | 4144.55 | 4180.50 | 4165.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-10 14:45:00 | 4142.45 | 4180.50 | 4165.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 3874.05 | 4112.73 | 4136.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 3820.60 | 3933.27 | 4027.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 11:15:00 | 3807.70 | 3806.28 | 3875.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-16 12:00:00 | 3807.70 | 3806.28 | 3875.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 3805.30 | 3781.02 | 3797.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 15:00:00 | 3805.30 | 3781.02 | 3797.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 3797.00 | 3784.21 | 3797.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 3845.00 | 3784.21 | 3797.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 3823.10 | 3791.99 | 3800.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:30:00 | 3855.40 | 3791.99 | 3800.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 3839.45 | 3801.48 | 3803.68 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 11:15:00 | 3837.20 | 3808.63 | 3806.73 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-22 10:15:00 | 3780.00 | 3806.70 | 3808.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-22 12:15:00 | 3778.25 | 3796.74 | 3802.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 09:15:00 | 3771.90 | 3770.30 | 3780.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 09:15:00 | 3771.90 | 3770.30 | 3780.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 09:15:00 | 3771.90 | 3770.30 | 3780.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-24 11:00:00 | 3750.10 | 3766.26 | 3777.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-06 09:15:00 | 3690.05 | 3658.59 | 3655.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 09:15:00 | 3690.05 | 3658.59 | 3655.63 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-09-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 15:15:00 | 3634.00 | 3656.71 | 3658.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-08 10:15:00 | 3629.10 | 3649.44 | 3654.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 09:15:00 | 3665.10 | 3638.54 | 3645.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 09:15:00 | 3665.10 | 3638.54 | 3645.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 09:15:00 | 3665.10 | 3638.54 | 3645.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 09:45:00 | 3664.55 | 3638.54 | 3645.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 10:15:00 | 3647.95 | 3640.42 | 3645.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-11 11:15:00 | 3638.05 | 3640.42 | 3645.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 09:30:00 | 3644.10 | 3636.16 | 3639.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-12 11:30:00 | 3642.95 | 3635.22 | 3638.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 09:15:00 | 3674.20 | 3643.42 | 3641.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-13 09:15:00 | 3674.20 | 3643.42 | 3641.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-13 11:15:00 | 3700.35 | 3660.34 | 3649.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 10:15:00 | 3706.00 | 3711.37 | 3684.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-14 11:00:00 | 3706.00 | 3711.37 | 3684.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 13:15:00 | 3698.75 | 3710.22 | 3691.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 13:30:00 | 3679.10 | 3710.22 | 3691.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 14:15:00 | 3712.30 | 3710.64 | 3693.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-14 14:30:00 | 3693.40 | 3710.64 | 3693.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 15:15:00 | 3728.05 | 3714.12 | 3696.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:15:00 | 3707.00 | 3714.12 | 3696.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 3728.45 | 3716.99 | 3699.21 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-15 15:15:00 | 3679.95 | 3696.63 | 3696.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 09:15:00 | 3662.95 | 3689.90 | 3693.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 10:15:00 | 3667.55 | 3664.15 | 3674.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-20 11:00:00 | 3667.55 | 3664.15 | 3674.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 12:15:00 | 3690.00 | 3669.32 | 3675.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 12:30:00 | 3684.90 | 3669.32 | 3675.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 13:15:00 | 3686.00 | 3672.66 | 3676.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 13:30:00 | 3684.75 | 3672.66 | 3676.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 13:15:00 | 3670.30 | 3666.67 | 3671.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 14:00:00 | 3670.30 | 3666.67 | 3671.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 3652.25 | 3663.78 | 3669.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 3635.00 | 3661.03 | 3667.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 10:15:00 | 3611.70 | 3551.95 | 3550.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 3611.70 | 3551.95 | 3550.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 10:15:00 | 3640.00 | 3604.18 | 3582.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 13:15:00 | 3577.00 | 3605.91 | 3589.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 13:15:00 | 3577.00 | 3605.91 | 3589.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 13:15:00 | 3577.00 | 3605.91 | 3589.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 14:00:00 | 3577.00 | 3605.91 | 3589.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 14:15:00 | 3572.65 | 3599.26 | 3587.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 15:00:00 | 3572.65 | 3599.26 | 3587.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 15:15:00 | 3568.50 | 3593.11 | 3586.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-04 09:15:00 | 3555.10 | 3593.11 | 3586.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 10:15:00 | 3522.10 | 3573.27 | 3577.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 3502.00 | 3559.02 | 3570.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 10:15:00 | 3509.90 | 3507.11 | 3533.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-05 10:45:00 | 3512.00 | 3507.11 | 3533.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 3495.00 | 3482.96 | 3508.28 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-09 11:15:00 | 3520.00 | 3515.53 | 3515.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-09 12:15:00 | 3525.30 | 3517.48 | 3516.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 11:15:00 | 3578.90 | 3580.32 | 3560.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 12:00:00 | 3578.90 | 3580.32 | 3560.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 3554.15 | 3575.09 | 3560.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:00:00 | 3554.15 | 3575.09 | 3560.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 3540.65 | 3568.20 | 3558.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 3541.00 | 3568.20 | 3558.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 3639.10 | 3593.32 | 3576.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 11:15:00 | 3647.10 | 3619.74 | 3610.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-20 09:15:00 | 3590.00 | 3610.46 | 3612.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 09:15:00 | 3590.00 | 3610.46 | 3612.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 3552.25 | 3589.69 | 3602.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 15:15:00 | 3586.10 | 3585.71 | 3596.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-23 09:15:00 | 3593.75 | 3585.71 | 3596.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 3585.95 | 3585.76 | 3595.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 11:30:00 | 3566.45 | 3586.19 | 3594.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 14:30:00 | 3562.35 | 3574.90 | 3587.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-25 11:15:00 | 3565.95 | 3565.61 | 3579.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-26 09:15:00 | 3552.25 | 3568.87 | 3575.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 14:15:00 | 3561.20 | 3553.11 | 3562.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-26 15:00:00 | 3561.20 | 3553.11 | 3562.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 15:15:00 | 3570.00 | 3556.49 | 3562.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:15:00 | 3602.05 | 3556.49 | 3562.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 3624.25 | 3570.04 | 3568.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 09:15:00 | 3624.25 | 3570.04 | 3568.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 11:15:00 | 3646.70 | 3597.41 | 3581.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 3692.85 | 3705.74 | 3680.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 3692.85 | 3705.74 | 3680.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 3692.85 | 3705.74 | 3680.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 3692.85 | 3705.74 | 3680.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 3809.15 | 3817.16 | 3795.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-06 13:45:00 | 3795.00 | 3817.16 | 3795.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 4389.80 | 4405.59 | 4384.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 12:30:00 | 4389.95 | 4405.59 | 4384.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 13:15:00 | 4389.65 | 4402.40 | 4384.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 13:45:00 | 4386.95 | 4402.40 | 4384.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 14:15:00 | 4389.85 | 4399.89 | 4385.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 14:30:00 | 4389.85 | 4399.89 | 4385.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-28 10:15:00 | 4544.95 | 4553.39 | 4534.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-28 11:00:00 | 4544.95 | 4553.39 | 4534.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 4635.00 | 4652.97 | 4630.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 14:45:00 | 4635.50 | 4652.97 | 4630.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 4645.00 | 4651.38 | 4631.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 09:15:00 | 4623.15 | 4651.38 | 4631.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 4639.60 | 4649.02 | 4632.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 09:30:00 | 4643.75 | 4649.02 | 4632.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 10:15:00 | 4609.90 | 4641.20 | 4630.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-04 11:00:00 | 4609.90 | 4641.20 | 4630.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 11:15:00 | 4615.35 | 4636.03 | 4628.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 13:45:00 | 4644.70 | 4630.25 | 4627.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-04 15:15:00 | 4615.00 | 4624.12 | 4624.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2023-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 15:15:00 | 4615.00 | 4624.12 | 4624.75 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 09:15:00 | 4642.55 | 4627.80 | 4626.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 10:15:00 | 4656.55 | 4633.55 | 4629.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 11:15:00 | 4599.95 | 4626.83 | 4626.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 4599.95 | 4626.83 | 4626.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 4599.95 | 4626.83 | 4626.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 11:45:00 | 4599.95 | 4626.83 | 4626.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 4600.00 | 4621.47 | 4624.06 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 15:15:00 | 4630.10 | 4625.11 | 4624.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 09:15:00 | 4745.00 | 4649.08 | 4635.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 10:15:00 | 4794.00 | 4827.26 | 4775.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-08 11:00:00 | 4794.00 | 4827.26 | 4775.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 11:15:00 | 4785.40 | 4818.89 | 4776.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-08 15:15:00 | 4799.00 | 4800.57 | 4777.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 09:15:00 | 4729.90 | 4786.19 | 4775.06 | SL hit (close<static) qty=1.00 sl=4773.65 alert=retest2 |

### Cycle 34 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 4708.95 | 4756.78 | 4762.76 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2023-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 12:15:00 | 4784.75 | 4755.17 | 4754.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 09:15:00 | 4847.65 | 4789.22 | 4771.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 13:15:00 | 4866.95 | 4870.10 | 4838.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 13:45:00 | 4866.95 | 4870.10 | 4838.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 14:15:00 | 4912.15 | 4906.70 | 4876.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-15 15:00:00 | 4912.15 | 4906.70 | 4876.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 15:15:00 | 4943.40 | 4914.04 | 4882.60 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 4870.45 | 4918.57 | 4919.20 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 4934.00 | 4918.94 | 4917.36 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 10:15:00 | 4902.20 | 4915.59 | 4915.98 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2023-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 11:15:00 | 4938.80 | 4920.23 | 4918.05 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 4867.90 | 4909.77 | 4913.50 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 4967.05 | 4919.97 | 4916.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 5028.75 | 4970.59 | 4946.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 09:15:00 | 5032.40 | 5037.93 | 4999.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 10:00:00 | 5032.40 | 5037.93 | 4999.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 5151.35 | 5161.18 | 5127.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:30:00 | 5133.75 | 5161.18 | 5127.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 5119.90 | 5152.93 | 5127.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 5119.90 | 5152.93 | 5127.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 5120.10 | 5146.36 | 5126.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 5161.40 | 5146.36 | 5126.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 10:15:00 | 5174.95 | 5228.95 | 5231.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 10:15:00 | 5174.95 | 5228.95 | 5231.14 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 5250.00 | 5223.39 | 5222.74 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 12:15:00 | 5196.40 | 5217.99 | 5220.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 13:15:00 | 5187.95 | 5211.98 | 5217.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 11:15:00 | 5052.20 | 5052.16 | 5091.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-12 12:00:00 | 5052.20 | 5052.16 | 5091.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 5051.10 | 5036.83 | 5053.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:45:00 | 5049.50 | 5036.83 | 5053.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 5053.45 | 5040.15 | 5053.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:15:00 | 5054.30 | 5040.15 | 5053.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 5047.55 | 5041.63 | 5052.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 11:45:00 | 5065.00 | 5041.63 | 5052.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 12:15:00 | 5040.00 | 5041.30 | 5051.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:45:00 | 5051.35 | 5041.30 | 5051.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 14:15:00 | 5062.00 | 5046.04 | 5051.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-16 15:00:00 | 5062.00 | 5046.04 | 5051.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 5055.00 | 5047.83 | 5052.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-17 09:15:00 | 5048.65 | 5047.83 | 5052.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-17 09:15:00 | 5068.70 | 5052.01 | 5053.71 | SL hit (close>static) qty=1.00 sl=5068.40 alert=retest2 |

### Cycle 45 — BUY (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-17 11:15:00 | 5060.00 | 5055.27 | 5055.00 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 15:15:00 | 5046.35 | 5054.91 | 5055.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 4941.30 | 5032.18 | 5044.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 14:15:00 | 4965.50 | 4948.69 | 4973.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 15:00:00 | 4965.50 | 4948.69 | 4973.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 09:15:00 | 4924.00 | 4943.09 | 4966.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-20 10:30:00 | 4920.65 | 4932.06 | 4959.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-23 09:45:00 | 4906.00 | 4890.59 | 4924.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 09:45:00 | 4918.65 | 4856.62 | 4883.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:15:00 | 4918.50 | 4888.58 | 4894.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-24 12:15:00 | 4939.55 | 4898.78 | 4898.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 4939.55 | 4898.78 | 4898.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 4983.35 | 4915.69 | 4906.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 13:15:00 | 4981.05 | 4987.32 | 4955.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-25 13:45:00 | 4989.00 | 4987.32 | 4955.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 14:15:00 | 4970.00 | 4983.86 | 4956.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 15:00:00 | 4970.00 | 4983.86 | 4956.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 4942.00 | 4975.49 | 4955.41 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2024-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 10:15:00 | 4923.00 | 4953.82 | 4954.90 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 13:15:00 | 4985.00 | 4956.00 | 4954.99 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 14:15:00 | 4942.65 | 4953.33 | 4953.87 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 15:15:00 | 4971.50 | 4956.96 | 4955.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 5059.00 | 4977.37 | 4964.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-31 14:15:00 | 4992.25 | 5006.58 | 4987.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-31 14:45:00 | 4993.95 | 5006.58 | 4987.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 15:15:00 | 4998.95 | 5005.05 | 4988.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-01 09:15:00 | 5033.05 | 5005.05 | 4988.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-01 10:15:00 | 4970.15 | 4997.70 | 4987.97 | SL hit (close<static) qty=1.00 sl=4981.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 12:15:00 | 4902.05 | 4968.81 | 4975.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 13:15:00 | 4866.60 | 4948.37 | 4965.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 10:15:00 | 4913.25 | 4904.68 | 4935.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-02 11:00:00 | 4913.25 | 4904.68 | 4935.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 14:15:00 | 4920.00 | 4909.24 | 4928.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 14:30:00 | 4920.30 | 4909.24 | 4928.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 15:15:00 | 4920.00 | 4911.39 | 4927.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 09:15:00 | 4992.35 | 4911.39 | 4927.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 09:15:00 | 5037.15 | 4936.54 | 4937.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-05 10:00:00 | 5037.15 | 4936.54 | 4937.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2024-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 10:15:00 | 5075.90 | 4964.41 | 4950.02 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 15:15:00 | 4915.80 | 4947.12 | 4948.09 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 09:15:00 | 5050.00 | 4967.70 | 4957.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-06 13:15:00 | 5110.35 | 5002.39 | 4977.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 09:15:00 | 5147.00 | 5280.05 | 5237.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 5147.00 | 5280.05 | 5237.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 5147.00 | 5280.05 | 5237.43 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 5107.00 | 5200.54 | 5207.69 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2024-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 10:15:00 | 5275.55 | 5206.24 | 5204.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-13 12:15:00 | 5290.55 | 5231.77 | 5217.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-14 09:15:00 | 5180.55 | 5246.83 | 5231.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 09:15:00 | 5180.55 | 5246.83 | 5231.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 5180.55 | 5246.83 | 5231.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:45:00 | 5159.75 | 5246.83 | 5231.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 10:15:00 | 5155.40 | 5228.55 | 5224.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-14 11:00:00 | 5155.40 | 5228.55 | 5224.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 11:15:00 | 5185.55 | 5219.95 | 5221.35 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 5244.00 | 5223.38 | 5221.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 12:15:00 | 5288.00 | 5243.30 | 5231.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 5377.85 | 5438.50 | 5393.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 5377.85 | 5438.50 | 5393.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 5377.85 | 5438.50 | 5393.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 5377.85 | 5438.50 | 5393.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 5374.05 | 5425.61 | 5392.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:00:00 | 5374.05 | 5425.61 | 5392.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 5391.45 | 5418.78 | 5392.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 12:15:00 | 5400.30 | 5418.78 | 5392.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:00:00 | 5407.95 | 5413.46 | 5396.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 14:15:00 | 5396.90 | 5424.56 | 5421.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 14:45:00 | 5414.00 | 5423.10 | 5421.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 5427.00 | 5423.88 | 5421.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 5428.35 | 5423.88 | 5421.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 5429.95 | 5425.09 | 5422.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 10:15:00 | 5383.85 | 5425.09 | 5422.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-23 10:15:00 | 5382.80 | 5416.64 | 5418.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2024-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 10:15:00 | 5382.80 | 5416.64 | 5418.77 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-02-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 12:15:00 | 5455.15 | 5427.16 | 5423.36 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 15:15:00 | 5382.00 | 5421.30 | 5421.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 09:15:00 | 5295.95 | 5396.23 | 5410.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 09:15:00 | 5032.45 | 5016.74 | 5076.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 09:30:00 | 5039.85 | 5016.74 | 5076.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 5068.90 | 5030.84 | 5072.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:30:00 | 5068.90 | 5030.84 | 5072.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 5087.75 | 5047.53 | 5073.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:45:00 | 5086.55 | 5047.53 | 5073.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 5138.60 | 5065.74 | 5079.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 5138.60 | 5065.74 | 5079.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-03-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 10:15:00 | 5117.95 | 5090.74 | 5088.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 10:15:00 | 5155.00 | 5125.52 | 5112.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 12:15:00 | 5121.05 | 5129.99 | 5116.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 12:15:00 | 5121.05 | 5129.99 | 5116.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 12:15:00 | 5121.05 | 5129.99 | 5116.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 13:00:00 | 5121.05 | 5129.99 | 5116.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 13:15:00 | 5132.50 | 5130.49 | 5118.18 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 10:15:00 | 5068.40 | 5107.32 | 5110.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 11:15:00 | 5059.60 | 5097.77 | 5105.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 14:15:00 | 5090.65 | 5085.39 | 5097.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 14:15:00 | 5090.65 | 5085.39 | 5097.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 5090.65 | 5085.39 | 5097.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 15:00:00 | 5090.65 | 5085.39 | 5097.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 15:15:00 | 5096.00 | 5087.51 | 5097.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:15:00 | 5154.00 | 5087.51 | 5097.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 5143.80 | 5098.77 | 5101.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 09:45:00 | 5120.00 | 5098.77 | 5101.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-06 10:15:00 | 5134.25 | 5105.87 | 5104.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-06 13:15:00 | 5187.95 | 5131.59 | 5117.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-07 11:15:00 | 5146.30 | 5159.86 | 5140.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-07 12:00:00 | 5146.30 | 5159.86 | 5140.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 14:15:00 | 5144.95 | 5157.19 | 5143.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-07 15:00:00 | 5144.95 | 5157.19 | 5143.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 15:15:00 | 5149.00 | 5155.55 | 5144.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-11 09:15:00 | 5178.55 | 5155.55 | 5144.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 09:15:00 | 5132.05 | 5155.76 | 5153.45 | SL hit (close<static) qty=1.00 sl=5137.60 alert=retest2 |

### Cycle 66 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 5134.95 | 5151.60 | 5151.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 5076.85 | 5136.65 | 5144.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 4996.00 | 4995.08 | 5044.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 09:45:00 | 4993.20 | 4995.08 | 5044.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 09:15:00 | 4913.40 | 4968.17 | 5006.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 10:15:00 | 4902.90 | 4968.17 | 5006.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-19 10:15:00 | 4902.00 | 4957.04 | 4969.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-26 09:15:00 | 4657.75 | 4790.68 | 4808.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-03-26 09:15:00 | 4792.90 | 4790.68 | 4808.53 | SL hit (close>static) qty=0.50 sl=4790.68 alert=retest2 |

### Cycle 67 — BUY (started 2024-03-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 11:15:00 | 4933.10 | 4831.74 | 4824.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 12:15:00 | 4982.50 | 4861.89 | 4839.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-27 14:15:00 | 5001.10 | 5001.15 | 4943.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-27 15:00:00 | 5001.10 | 5001.15 | 4943.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 4855.50 | 4972.47 | 4940.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 12:15:00 | 4946.00 | 4953.69 | 4936.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-28 15:15:00 | 4949.80 | 4957.90 | 4943.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 10:00:00 | 4989.35 | 4981.12 | 4965.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-02 14:30:00 | 4940.55 | 4966.24 | 4964.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-03 09:15:00 | 4925.00 | 4956.19 | 4959.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 09:15:00 | 4925.00 | 4956.19 | 4959.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 10:15:00 | 4902.50 | 4945.46 | 4954.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 15:15:00 | 4886.00 | 4870.56 | 4896.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-05 09:15:00 | 4983.40 | 4870.56 | 4896.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 5044.55 | 4905.36 | 4909.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-05 10:00:00 | 5044.55 | 4905.36 | 4909.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 4966.60 | 4917.61 | 4914.93 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 4871.55 | 4907.31 | 4912.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 4860.00 | 4883.32 | 4897.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-08 14:15:00 | 4887.40 | 4884.13 | 4896.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-08 14:45:00 | 4886.20 | 4884.13 | 4896.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 15:15:00 | 4884.00 | 4884.11 | 4895.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 09:15:00 | 4866.45 | 4884.11 | 4895.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 4910.00 | 4889.29 | 4897.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 10:00:00 | 4910.00 | 4889.29 | 4897.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 10:15:00 | 4878.30 | 4887.09 | 4895.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 12:15:00 | 4870.20 | 4885.55 | 4893.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 12:45:00 | 4875.45 | 4882.44 | 4891.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 15:15:00 | 4867.50 | 4881.34 | 4889.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 4626.69 | 4672.99 | 4707.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 4631.68 | 4672.99 | 4707.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-19 09:15:00 | 4624.12 | 4672.99 | 4707.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 4648.85 | 4630.92 | 4662.29 | SL hit (close>ema200) qty=0.50 sl=4630.92 alert=retest2 |

### Cycle 71 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 4741.75 | 4682.81 | 4680.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 11:15:00 | 4765.40 | 4736.93 | 4723.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 13:15:00 | 4910.00 | 4917.08 | 4871.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 13:45:00 | 4927.65 | 4917.08 | 4871.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 14:15:00 | 4882.00 | 4910.07 | 4872.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 14:45:00 | 4879.85 | 4910.07 | 4872.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 4887.00 | 4905.45 | 4873.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:15:00 | 4937.40 | 4905.45 | 4873.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 12:15:00 | 4864.00 | 4898.16 | 4880.92 | SL hit (close<static) qty=1.00 sl=4870.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 15:15:00 | 4825.00 | 4869.11 | 4870.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 4759.55 | 4847.20 | 4860.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-03 12:15:00 | 4803.30 | 4800.05 | 4819.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-03 12:15:00 | 4803.30 | 4800.05 | 4819.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 12:15:00 | 4803.30 | 4800.05 | 4819.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 12:30:00 | 4822.90 | 4800.05 | 4819.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 4902.40 | 4822.65 | 4826.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-03 15:00:00 | 4902.40 | 4822.65 | 4826.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 15:15:00 | 4898.85 | 4837.89 | 4832.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 10:15:00 | 4999.15 | 4880.08 | 4853.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 5050.00 | 5055.41 | 4974.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 11:00:00 | 5050.00 | 5055.41 | 4974.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 12:15:00 | 5078.50 | 5113.61 | 5085.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 13:00:00 | 5078.50 | 5113.61 | 5085.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 13:15:00 | 5117.85 | 5114.46 | 5088.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 10:30:00 | 5131.05 | 5101.61 | 5088.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 11:30:00 | 5151.00 | 5107.11 | 5092.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 5179.60 | 5113.95 | 5100.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 10:15:00 | 5127.00 | 5113.63 | 5101.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 10:15:00 | 5144.55 | 5119.81 | 5105.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-13 10:30:00 | 5109.65 | 5119.81 | 5105.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 5114.90 | 5163.54 | 5139.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:00:00 | 5114.90 | 5163.54 | 5139.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 5154.10 | 5161.65 | 5140.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 10:45:00 | 5146.95 | 5161.65 | 5140.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 5210.05 | 5228.64 | 5203.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 09:45:00 | 5191.10 | 5228.64 | 5203.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 5276.45 | 5238.21 | 5210.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:45:00 | 5329.95 | 5291.26 | 5252.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 11:00:00 | 5363.00 | 5397.84 | 5353.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 12:30:00 | 5337.45 | 5383.77 | 5354.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 15:00:00 | 5330.20 | 5375.49 | 5355.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 5335.15 | 5367.42 | 5353.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 5295.75 | 5367.42 | 5353.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 5237.70 | 5341.48 | 5343.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 5237.70 | 5341.48 | 5343.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 5230.50 | 5306.11 | 5325.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 5306.90 | 5298.60 | 5316.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:00:00 | 5306.90 | 5298.60 | 5316.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 5285.10 | 5295.90 | 5313.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 5296.60 | 5295.90 | 5313.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 5285.40 | 5293.80 | 5311.29 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 5427.10 | 5321.96 | 5321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 5447.95 | 5347.15 | 5332.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 5472.80 | 5416.75 | 5379.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 13:00:00 | 5470.00 | 5427.40 | 5387.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 5325.50 | 5390.86 | 5389.43 | SL hit (close<static) qty=1.00 sl=5355.10 alert=retest2 |

### Cycle 76 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 5329.00 | 5378.49 | 5383.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 5270.85 | 5315.66 | 5333.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 4771.15 | 4740.30 | 4827.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 14:00:00 | 4771.15 | 4740.30 | 4827.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 4835.60 | 4754.07 | 4811.82 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 4863.05 | 4824.90 | 4822.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 4870.15 | 4833.95 | 4826.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 5060.20 | 5072.30 | 5044.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 5060.20 | 5072.30 | 5044.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 5084.80 | 5081.74 | 5060.93 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 4943.75 | 5040.47 | 5053.37 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 5214.40 | 5084.20 | 5068.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 12:15:00 | 5251.90 | 5175.15 | 5125.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 5178.95 | 5181.99 | 5137.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 5178.95 | 5181.99 | 5137.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 5148.25 | 5176.41 | 5142.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:45:00 | 5186.50 | 5180.05 | 5149.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 5127.90 | 5157.80 | 5152.51 | SL hit (close<static) qty=1.00 sl=5137.70 alert=retest2 |

### Cycle 80 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 5126.90 | 5145.81 | 5147.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 5104.75 | 5137.60 | 5143.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 5006.15 | 5081.24 | 5087.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:00:00 | 5027.15 | 5057.62 | 5072.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 5029.00 | 4961.27 | 4958.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 5029.00 | 4961.27 | 4958.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 5071.95 | 5007.81 | 4989.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 5232.25 | 5278.57 | 5208.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 5232.25 | 5278.57 | 5208.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 5184.90 | 5259.84 | 5206.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 5200.10 | 5259.84 | 5206.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 5196.95 | 5247.26 | 5205.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 5225.00 | 5241.88 | 5206.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:45:00 | 5220.20 | 5237.10 | 5207.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 5221.55 | 5237.10 | 5207.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:30:00 | 5231.15 | 5222.43 | 5215.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 5208.85 | 5219.71 | 5215.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 5252.95 | 5219.71 | 5215.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 5212.85 | 5214.78 | 5213.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 5196.75 | 5211.17 | 5212.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 5196.75 | 5211.17 | 5212.09 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 5380.00 | 5238.23 | 5223.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 5408.10 | 5293.38 | 5252.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 5338.65 | 5357.58 | 5312.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 11:45:00 | 5342.10 | 5357.58 | 5312.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 5304.95 | 5341.83 | 5313.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:30:00 | 5313.10 | 5341.83 | 5313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 5300.20 | 5333.51 | 5311.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 5296.55 | 5333.51 | 5311.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 5294.00 | 5325.61 | 5310.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 5285.00 | 5325.61 | 5310.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 5234.55 | 5296.48 | 5299.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 5192.70 | 5251.41 | 5272.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 5216.65 | 5188.05 | 5222.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 5216.65 | 5188.05 | 5222.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 5185.00 | 5187.44 | 5219.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 5203.40 | 5187.44 | 5219.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 5220.00 | 5195.84 | 5217.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 5217.75 | 5195.84 | 5217.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 5222.10 | 5201.09 | 5218.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 5232.20 | 5201.09 | 5218.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 5230.00 | 5209.60 | 5219.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 5199.15 | 5209.60 | 5219.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 5218.75 | 5211.43 | 5219.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 5233.00 | 5211.43 | 5219.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 5197.35 | 5208.62 | 5217.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 5111.55 | 5207.95 | 5216.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 5157.65 | 5201.00 | 5211.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 5231.15 | 5204.04 | 5209.68 | SL hit (close>static) qty=1.00 sl=5218.75 alert=retest2 |

### Cycle 85 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 5259.55 | 5215.14 | 5214.22 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 14:15:00 | 5199.05 | 5213.41 | 5214.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 5153.85 | 5197.07 | 5206.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 5196.90 | 5192.77 | 5201.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 5173.05 | 5188.82 | 5199.25 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 5293.10 | 5207.06 | 5204.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 5314.00 | 5269.14 | 5241.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 10:15:00 | 5268.05 | 5272.42 | 5247.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 10:45:00 | 5278.70 | 5272.42 | 5247.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 5234.40 | 5264.82 | 5246.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 5234.40 | 5264.82 | 5246.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 5228.20 | 5257.49 | 5244.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:15:00 | 5226.10 | 5257.49 | 5244.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 5208.45 | 5247.68 | 5241.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 5208.45 | 5247.68 | 5241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 5193.60 | 5236.87 | 5237.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 5187.00 | 5222.44 | 5230.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:45:00 | 5227.15 | 5221.16 | 5226.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 5221.70 | 5221.26 | 5226.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 5205.00 | 5221.26 | 5226.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 5290.00 | 5232.41 | 5230.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 5290.00 | 5232.41 | 5230.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 13:15:00 | 5333.05 | 5273.90 | 5252.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 5265.25 | 5289.21 | 5269.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 5310.10 | 5293.38 | 5273.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:45:00 | 5324.80 | 5293.30 | 5281.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:15:00 | 5323.65 | 5293.30 | 5281.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 5255.90 | 5289.30 | 5283.86 | SL hit (close<static) qty=1.00 sl=5257.70 alert=retest2 |

### Cycle 90 — SELL (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 09:15:00 | 5650.60 | 5692.97 | 5697.19 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 5726.10 | 5689.45 | 5688.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 5745.25 | 5700.61 | 5694.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 5712.05 | 5722.28 | 5709.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 5712.05 | 5722.28 | 5709.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 5738.45 | 5729.24 | 5714.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 5738.45 | 5729.24 | 5714.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 5732.55 | 5729.02 | 5717.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:30:00 | 5761.90 | 5739.24 | 5724.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 5776.30 | 5764.35 | 5761.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-05 10:15:00 | 6338.09 | 6215.74 | 6167.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 6267.45 | 6343.96 | 6346.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 6221.30 | 6306.78 | 6328.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 6143.25 | 6139.62 | 6201.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 12:45:00 | 6115.70 | 6132.10 | 6182.36 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 6124.20 | 6129.59 | 6164.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 6055.70 | 6119.39 | 6147.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:00:00 | 6093.35 | 6110.99 | 6138.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 6071.50 | 6031.70 | 6069.55 | SL hit (close>ema400) qty=1.00 sl=6069.55 alert=retest1 |

### Cycle 93 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 6110.05 | 6073.08 | 6072.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 12:15:00 | 6135.00 | 6085.46 | 6078.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 6146.05 | 6150.74 | 6116.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 6146.05 | 6150.74 | 6116.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 6060.00 | 6128.79 | 6112.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 6060.00 | 6128.79 | 6112.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 6063.30 | 6115.69 | 6107.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 6063.30 | 6115.69 | 6107.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 6200.45 | 6133.85 | 6117.61 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 6089.00 | 6132.04 | 6135.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 6051.00 | 6115.83 | 6127.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 6104.80 | 6094.40 | 6110.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:30:00 | 6105.65 | 6094.40 | 6110.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 6147.35 | 6104.99 | 6113.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 6147.35 | 6104.99 | 6113.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 6157.25 | 6115.44 | 6117.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 6157.25 | 6115.44 | 6117.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 6160.15 | 6123.02 | 6120.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 6234.30 | 6173.08 | 6150.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 6183.00 | 6210.84 | 6176.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 6200.00 | 6208.67 | 6178.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 6156.30 | 6208.67 | 6178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 6150.60 | 6197.06 | 6176.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 6150.60 | 6197.06 | 6176.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 6198.95 | 6197.44 | 6178.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:30:00 | 6120.00 | 6197.44 | 6178.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 6183.90 | 6194.73 | 6178.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:00:00 | 6183.90 | 6194.73 | 6178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 6128.10 | 6181.40 | 6174.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 6128.10 | 6181.40 | 6174.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 6128.55 | 6170.83 | 6170.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:45:00 | 6144.00 | 6170.83 | 6170.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 6147.35 | 6166.14 | 6167.99 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 6251.15 | 6177.05 | 6171.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 6260.00 | 6214.17 | 6192.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 6258.05 | 6280.98 | 6250.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:00:00 | 6258.05 | 6280.98 | 6250.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 6220.05 | 6268.79 | 6248.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 6220.05 | 6268.79 | 6248.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 6143.30 | 6243.69 | 6238.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 6143.30 | 6243.69 | 6238.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 6096.10 | 6214.17 | 6225.59 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 6253.85 | 6205.45 | 6201.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 6261.80 | 6223.80 | 6211.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 15:15:00 | 6251.05 | 6254.35 | 6235.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:15:00 | 6217.35 | 6254.35 | 6235.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 6134.60 | 6230.40 | 6226.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 6134.60 | 6230.40 | 6226.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 6125.10 | 6209.34 | 6217.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 6059.10 | 6179.29 | 6202.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:30:00 | 6105.00 | 6060.07 | 6105.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 6136.25 | 6073.08 | 6096.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 6151.75 | 6073.08 | 6096.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 6129.30 | 6084.32 | 6099.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:15:00 | 6154.00 | 6084.32 | 6099.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 6234.25 | 6133.22 | 6120.03 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 6111.05 | 6133.02 | 6134.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 11:15:00 | 6076.90 | 6119.00 | 6128.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 5955.95 | 5954.67 | 6007.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 5955.95 | 5954.67 | 6007.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 5947.10 | 5951.00 | 5996.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 5908.95 | 5940.45 | 5977.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 6016.70 | 5961.27 | 5975.82 | SL hit (close>static) qty=1.00 sl=6008.25 alert=retest2 |

### Cycle 103 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 6034.55 | 5985.32 | 5984.81 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 5891.80 | 5988.14 | 5989.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 5873.60 | 5965.23 | 5978.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 5925.00 | 5914.62 | 5946.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 15:00:00 | 5925.00 | 5914.62 | 5946.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 5910.40 | 5917.38 | 5942.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:30:00 | 5886.60 | 5903.78 | 5933.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 5828.55 | 5757.27 | 5753.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 5828.55 | 5757.27 | 5753.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 5849.85 | 5775.78 | 5762.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 5738.15 | 5780.69 | 5767.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 5734.95 | 5771.54 | 5764.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 5717.50 | 5771.54 | 5764.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 5723.30 | 5761.89 | 5761.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 5723.30 | 5761.89 | 5761.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 5695.70 | 5748.66 | 5755.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 5591.55 | 5684.76 | 5712.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 5510.70 | 5490.75 | 5561.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 5510.70 | 5490.75 | 5561.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 5586.00 | 5509.80 | 5563.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:45:00 | 5570.95 | 5509.80 | 5563.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 5564.05 | 5520.65 | 5563.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 10:00:00 | 5512.65 | 5526.17 | 5559.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:45:00 | 5516.90 | 5532.36 | 5556.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 5521.95 | 5537.51 | 5553.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 5519.50 | 5537.32 | 5550.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 5552.20 | 5540.30 | 5550.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:45:00 | 5572.00 | 5540.30 | 5550.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 5577.40 | 5547.72 | 5552.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 5577.40 | 5547.72 | 5552.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 5577.10 | 5553.60 | 5555.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 5580.65 | 5553.60 | 5555.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 5561.95 | 5540.70 | 5546.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 5561.95 | 5540.70 | 5546.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 5586.10 | 5549.78 | 5550.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 5585.15 | 5549.78 | 5550.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 5577.85 | 5555.39 | 5552.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 5577.85 | 5555.39 | 5552.60 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 5534.00 | 5547.97 | 5549.75 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 09:15:00 | 5609.65 | 5560.31 | 5555.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 5659.65 | 5580.18 | 5564.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 5553.15 | 5580.98 | 5568.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 5570.90 | 5578.97 | 5568.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:15:00 | 5586.35 | 5578.97 | 5568.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 5529.00 | 5570.56 | 5568.78 | SL hit (close<static) qty=1.00 sl=5541.55 alert=retest2 |

### Cycle 110 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 5536.40 | 5563.73 | 5565.84 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 5606.30 | 5569.55 | 5567.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5650.00 | 5585.64 | 5574.88 | Break + close above crossover candle high |

### Cycle 112 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 5482.60 | 5577.55 | 5577.66 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 5591.55 | 5498.21 | 5486.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 11:15:00 | 5609.30 | 5520.43 | 5497.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 5660.05 | 5674.44 | 5623.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 5660.05 | 5674.44 | 5623.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 5640.35 | 5661.46 | 5626.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 5637.95 | 5661.46 | 5626.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 5639.70 | 5657.11 | 5627.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:30:00 | 5628.20 | 5657.11 | 5627.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 5675.00 | 5663.37 | 5645.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 5660.50 | 5663.37 | 5645.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 5678.10 | 5666.31 | 5648.50 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 5571.25 | 5630.23 | 5637.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 10:15:00 | 5531.80 | 5610.55 | 5627.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 5511.55 | 5472.99 | 5501.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 5499.90 | 5478.37 | 5501.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:45:00 | 5498.35 | 5478.37 | 5501.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 5492.35 | 5481.17 | 5500.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:30:00 | 5498.50 | 5481.17 | 5500.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 5510.70 | 5487.07 | 5501.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:00:00 | 5510.70 | 5487.07 | 5501.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 5486.85 | 5487.03 | 5500.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 5449.30 | 5488.11 | 5498.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 5502.00 | 5414.41 | 5413.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 5502.00 | 5414.41 | 5413.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 5533.60 | 5438.25 | 5424.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:30:00 | 5502.45 | 5481.98 | 5458.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 5501.95 | 5498.99 | 5475.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 5504.60 | 5496.63 | 5477.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 13:15:00 | 5500.50 | 5500.95 | 5485.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 5505.20 | 5501.80 | 5487.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 15:15:00 | 5425.00 | 5473.15 | 5475.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 5425.00 | 5473.15 | 5475.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 15:15:00 | 5395.00 | 5438.72 | 5452.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 5398.20 | 5394.71 | 5421.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 14:00:00 | 5398.20 | 5394.71 | 5421.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 5414.00 | 5398.57 | 5421.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 5414.00 | 5398.57 | 5421.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 5419.95 | 5402.84 | 5421.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 5417.60 | 5402.84 | 5421.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 5471.65 | 5416.60 | 5425.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 5473.00 | 5416.60 | 5425.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 5464.90 | 5426.26 | 5429.28 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 5467.70 | 5434.55 | 5432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 5487.40 | 5445.12 | 5437.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 5523.75 | 5526.19 | 5491.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 14:00:00 | 5523.75 | 5526.19 | 5491.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 5575.90 | 5613.15 | 5585.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:45:00 | 5579.95 | 5613.15 | 5585.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 5569.40 | 5604.40 | 5584.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 5569.40 | 5604.40 | 5584.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 5572.00 | 5597.92 | 5583.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 5553.70 | 5597.92 | 5583.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 5533.55 | 5572.58 | 5573.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 5495.90 | 5556.59 | 5565.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 11:15:00 | 5515.00 | 5505.92 | 5527.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:30:00 | 5500.00 | 5505.92 | 5527.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 5516.35 | 5508.01 | 5526.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 5531.75 | 5508.01 | 5526.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 5525.50 | 5509.21 | 5523.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 5525.50 | 5509.21 | 5523.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 5527.00 | 5512.77 | 5523.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 5537.90 | 5512.77 | 5523.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 5583.65 | 5526.94 | 5529.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 5583.65 | 5526.94 | 5529.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 5640.65 | 5549.68 | 5539.33 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 5486.40 | 5539.14 | 5542.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 5454.50 | 5508.28 | 5523.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 5277.00 | 5273.45 | 5316.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 5203.00 | 5273.45 | 5316.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 5258.60 | 5208.67 | 5228.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 5258.60 | 5208.67 | 5228.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 5240.20 | 5214.98 | 5229.75 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 14:15:00 | 5249.80 | 5236.41 | 5236.16 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 5199.75 | 5231.85 | 5234.31 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 5256.40 | 5236.76 | 5236.31 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 5225.20 | 5234.45 | 5235.30 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 5246.60 | 5237.01 | 5236.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 15:15:00 | 5265.00 | 5242.61 | 5238.87 | Break + close above crossover candle high |

### Cycle 126 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 5208.00 | 5236.58 | 5236.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 5158.45 | 5210.63 | 5223.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 5224.50 | 5165.55 | 5183.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 5216.80 | 5175.80 | 5186.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 5215.85 | 5175.80 | 5186.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 5196.75 | 5186.98 | 5189.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 5196.75 | 5186.98 | 5189.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 5188.30 | 5187.24 | 5189.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 5166.05 | 5187.24 | 5189.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 4907.75 | 5027.77 | 5091.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 4930.80 | 4929.27 | 4975.09 | SL hit (close>ema200) qty=0.50 sl=4929.27 alert=retest2 |

### Cycle 127 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 5074.45 | 4994.48 | 4988.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 5088.80 | 5048.17 | 5037.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:00:00 | 5043.90 | 5062.29 | 5048.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 5044.55 | 5058.75 | 5048.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:45:00 | 5028.90 | 5058.75 | 5048.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 5050.00 | 5057.00 | 5048.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 5073.45 | 5057.00 | 5048.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:45:00 | 5061.55 | 5056.69 | 5049.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 5066.50 | 5056.69 | 5049.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 4829.45 | 5123.45 | 5153.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 4829.45 | 5123.45 | 5153.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 4712.30 | 4858.71 | 4980.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 4704.60 | 4689.05 | 4770.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 4704.60 | 4689.05 | 4770.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 4696.20 | 4690.48 | 4763.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 4624.95 | 4714.58 | 4742.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 4689.55 | 4658.95 | 4688.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 4754.10 | 4709.88 | 4706.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 13:15:00 | 4754.10 | 4709.88 | 4706.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 4800.90 | 4728.08 | 4714.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 4743.50 | 4747.48 | 4732.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 4721.15 | 4742.22 | 4731.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 4721.15 | 4742.22 | 4731.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 4711.70 | 4736.11 | 4729.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 4677.40 | 4736.11 | 4729.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 4745.35 | 4737.96 | 4731.36 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 4723.90 | 4731.87 | 4732.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 15:15:00 | 4709.15 | 4726.09 | 4729.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 4617.25 | 4613.39 | 4635.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:30:00 | 4616.10 | 4613.39 | 4635.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 4572.90 | 4583.53 | 4609.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:45:00 | 4510.20 | 4575.92 | 4593.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 4607.90 | 4593.57 | 4592.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 4607.90 | 4593.57 | 4592.92 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 09:15:00 | 4554.95 | 4590.74 | 4592.15 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 4601.35 | 4592.67 | 4592.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 4624.65 | 4599.06 | 4595.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 4600.00 | 4602.35 | 4597.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:15:00 | 4619.85 | 4602.35 | 4597.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4636.45 | 4609.17 | 4601.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:00:00 | 4649.15 | 4620.90 | 4608.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 14:45:00 | 4684.70 | 4639.50 | 4620.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 4624.00 | 4691.89 | 4699.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 4624.00 | 4691.89 | 4699.69 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 12:15:00 | 4725.65 | 4700.44 | 4699.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 4760.60 | 4712.47 | 4705.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 4715.00 | 4736.18 | 4722.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 4715.35 | 4732.02 | 4722.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 4710.95 | 4732.02 | 4722.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 4758.40 | 4734.88 | 4725.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 4763.00 | 4737.59 | 4728.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 4704.45 | 4725.35 | 4725.27 | SL hit (close<static) qty=1.00 sl=4713.35 alert=retest2 |

### Cycle 136 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 4681.40 | 4716.56 | 4721.28 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 4782.15 | 4729.68 | 4726.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 4844.95 | 4803.65 | 4775.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 5021.00 | 5036.81 | 4991.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:45:00 | 5014.60 | 5036.81 | 4991.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 4986.35 | 5026.72 | 4991.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 4995.35 | 5026.72 | 4991.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 4945.05 | 5010.39 | 4986.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 4945.05 | 5010.39 | 4986.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 5039.65 | 5016.24 | 4991.73 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 4953.50 | 4991.92 | 4993.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 4924.10 | 4978.36 | 4987.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 4959.40 | 4956.18 | 4971.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 4886.40 | 4939.00 | 4961.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 4841.05 | 4876.37 | 4908.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 4830.55 | 4859.11 | 4894.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 4830.00 | 4850.54 | 4884.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 4847.20 | 4849.87 | 4881.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 4885.00 | 4852.15 | 4871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 4885.00 | 4852.15 | 4871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 4903.00 | 4862.32 | 4874.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 4999.85 | 4862.32 | 4874.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 4965.50 | 4882.96 | 4882.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 4965.50 | 4882.96 | 4882.71 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 4770.75 | 4887.10 | 4890.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4661.85 | 4803.71 | 4841.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 4778.55 | 4734.34 | 4772.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 4817.70 | 4751.01 | 4776.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:45:00 | 4812.65 | 4751.01 | 4776.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 4838.55 | 4768.52 | 4782.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 4840.25 | 4768.52 | 4782.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 4837.15 | 4799.58 | 4794.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 4879.25 | 4847.95 | 4827.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 4853.40 | 4873.12 | 4851.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 4853.40 | 4873.12 | 4851.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 4878.00 | 4874.10 | 4853.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 14:00:00 | 4893.60 | 4878.41 | 4859.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 5041.20 | 5099.09 | 5106.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 5041.20 | 5099.09 | 5106.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 5016.10 | 5082.49 | 5098.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 5072.70 | 5070.24 | 5087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 5035.90 | 5021.70 | 5041.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 5035.90 | 5021.70 | 5041.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 5114.50 | 5041.62 | 5046.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 5108.50 | 5041.62 | 5046.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 5131.40 | 5059.58 | 5054.58 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 5061.00 | 5073.11 | 5073.80 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 5106.00 | 5078.52 | 5075.93 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 5053.00 | 5070.77 | 5072.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 5041.50 | 5064.92 | 5069.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 5034.00 | 5029.29 | 5045.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 5034.00 | 5029.29 | 5045.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 5027.50 | 5028.93 | 5043.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 5027.50 | 5028.93 | 5043.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 5044.00 | 5031.95 | 5043.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 5044.00 | 5031.95 | 5043.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 5035.50 | 5032.66 | 5043.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 5041.50 | 5032.66 | 5043.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 4948.00 | 4924.91 | 4958.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 4948.00 | 4924.91 | 4958.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 4950.00 | 4929.93 | 4958.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 4940.50 | 4929.93 | 4958.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5032.50 | 4950.45 | 4964.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 5032.50 | 4950.45 | 4964.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 5019.50 | 4964.26 | 4969.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 5006.00 | 4964.26 | 4969.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 5019.00 | 4975.20 | 4974.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 5019.00 | 4975.20 | 4974.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 5115.00 | 5017.31 | 4995.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 5086.00 | 5088.28 | 5049.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 5086.00 | 5088.28 | 5049.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 5071.50 | 5084.92 | 5051.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 5071.50 | 5084.92 | 5051.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5325.50 | 5248.18 | 5204.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:30:00 | 5391.50 | 5270.64 | 5218.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 5388.50 | 5307.79 | 5245.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 5377.00 | 5314.11 | 5291.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 5379.00 | 5314.11 | 5291.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 5321.00 | 5321.97 | 5301.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 5325.00 | 5321.97 | 5301.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5376.50 | 5351.89 | 5323.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 5294.00 | 5308.53 | 5310.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 5294.00 | 5308.53 | 5310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 5172.00 | 5229.31 | 5244.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 5293.50 | 5242.15 | 5249.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 5325.00 | 5258.72 | 5255.93 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 5098.50 | 5226.67 | 5241.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 5049.50 | 5114.56 | 5166.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 5096.00 | 5084.52 | 5128.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 5096.00 | 5084.52 | 5128.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 4995.00 | 5068.37 | 5114.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 4973.50 | 5041.70 | 5093.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 4980.00 | 5019.76 | 5069.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 4942.00 | 4894.09 | 4889.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 4942.00 | 4894.09 | 4889.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 4966.50 | 4917.68 | 4901.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 4918.50 | 4940.36 | 4923.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 4910.00 | 4934.29 | 4921.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 4918.50 | 4934.29 | 4921.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 4898.50 | 4913.13 | 4914.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 4870.50 | 4904.15 | 4909.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 4823.50 | 4815.04 | 4849.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 4823.50 | 4815.04 | 4849.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 4853.00 | 4822.63 | 4849.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 4853.00 | 4822.63 | 4849.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 4850.50 | 4828.21 | 4849.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 4832.00 | 4828.21 | 4849.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4810.00 | 4824.57 | 4846.25 | EMA400 retest candle locked (from downside) |

### Cycle 153 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 4867.00 | 4853.84 | 4852.17 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 4840.00 | 4850.85 | 4851.09 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 4855.00 | 4851.68 | 4851.44 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 4821.00 | 4845.55 | 4848.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 4791.00 | 4823.04 | 4836.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 4810.00 | 4807.77 | 4823.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:45:00 | 4814.00 | 4807.77 | 4823.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 4780.00 | 4800.97 | 4817.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 4763.00 | 4794.28 | 4813.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 4761.00 | 4781.39 | 4795.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:15:00 | 4763.00 | 4778.31 | 4792.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 4765.00 | 4775.95 | 4790.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4774.50 | 4775.66 | 4788.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 4774.50 | 4775.66 | 4788.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4774.50 | 4775.43 | 4787.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 4776.00 | 4775.43 | 4787.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 4767.50 | 4773.84 | 4785.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 4757.00 | 4776.41 | 4783.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 4800.00 | 4778.02 | 4782.44 | SL hit (close>static) qty=1.00 sl=4798.50 alert=retest2 |

### Cycle 157 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 4823.50 | 4789.19 | 4786.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 4847.00 | 4810.68 | 4799.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 4836.50 | 4840.81 | 4828.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 14:00:00 | 4836.50 | 4840.81 | 4828.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 4837.00 | 4840.05 | 4829.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 4829.00 | 4840.05 | 4829.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 4884.10 | 4917.15 | 4896.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 4884.10 | 4917.15 | 4896.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 4863.70 | 4906.46 | 4893.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 4863.70 | 4906.46 | 4893.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 4825.00 | 4877.63 | 4882.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 4819.00 | 4847.49 | 4859.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 4873.90 | 4843.46 | 4852.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 4866.70 | 4848.11 | 4853.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 4852.90 | 4851.01 | 4854.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 4858.70 | 4851.01 | 4854.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 4884.10 | 4857.62 | 4857.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 4884.10 | 4857.62 | 4857.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 4889.00 | 4863.90 | 4860.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 4851.60 | 4866.84 | 4862.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4860.10 | 4865.50 | 4862.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 4865.20 | 4864.40 | 4862.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 4813.80 | 4859.01 | 4861.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 4813.80 | 4859.01 | 4861.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 4801.40 | 4847.49 | 4856.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 4819.10 | 4816.19 | 4832.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 4822.10 | 4816.19 | 4832.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4797.10 | 4761.70 | 4783.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 4804.50 | 4761.70 | 4783.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 4777.10 | 4764.78 | 4783.22 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 4839.80 | 4796.59 | 4794.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 4843.30 | 4807.68 | 4800.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 4990.10 | 5005.13 | 4973.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 4990.10 | 5005.13 | 4973.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5006.80 | 5002.42 | 4977.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 4991.70 | 5002.42 | 4977.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 4987.40 | 4997.96 | 4984.81 | EMA400 retest candle locked (from upside) |

### Cycle 162 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4951.00 | 4985.28 | 4986.18 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 4991.90 | 4982.18 | 4982.07 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 4978.90 | 4981.53 | 4981.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 15:15:00 | 4967.60 | 4978.48 | 4980.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 5016.10 | 4978.87 | 4980.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 4993.70 | 4981.83 | 4981.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 12:15:00 | 5003.80 | 4988.30 | 4984.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 5013.20 | 5007.24 | 4999.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 5054.90 | 5016.77 | 5004.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 5071.00 | 5016.77 | 5004.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 5055.00 | 5039.80 | 5023.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 5055.90 | 5045.70 | 5029.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 5058.10 | 5061.47 | 5045.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5053.50 | 5059.87 | 5045.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 5040.60 | 5059.87 | 5045.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 5029.60 | 5072.32 | 5060.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 5029.60 | 5072.32 | 5060.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 5021.20 | 5062.10 | 5057.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 5038.00 | 5057.26 | 5055.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 5033.90 | 5050.30 | 5052.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 5033.90 | 5050.30 | 5052.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 5015.00 | 5043.24 | 5049.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 4895.00 | 4883.17 | 4929.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:45:00 | 4899.00 | 4883.17 | 4929.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 4857.00 | 4882.99 | 4918.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 4841.50 | 4890.49 | 4907.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 4850.00 | 4879.19 | 4901.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 4846.00 | 4859.02 | 4885.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 4840.00 | 4863.12 | 4884.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4891.00 | 4865.00 | 4881.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 4912.00 | 4865.00 | 4881.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 4855.50 | 4863.10 | 4879.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:00:00 | 4849.50 | 4860.38 | 4876.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 4844.00 | 4856.83 | 4868.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:45:00 | 4849.00 | 4814.91 | 4824.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 4928.50 | 4842.44 | 4835.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 4928.50 | 4842.44 | 4835.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 4959.50 | 4865.85 | 4846.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5341.50 | 5349.13 | 5292.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5341.50 | 5349.13 | 5292.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 5404.50 | 5361.95 | 5308.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 5439.00 | 5401.41 | 5377.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 5443.00 | 5417.34 | 5394.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 5405.00 | 5420.34 | 5421.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 5405.00 | 5420.34 | 5421.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 5378.00 | 5403.29 | 5411.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 5327.50 | 5311.48 | 5333.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 5251.00 | 5311.48 | 5333.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 5306.50 | 5310.49 | 5331.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:45:00 | 5245.50 | 5290.21 | 5318.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:00:00 | 5240.50 | 5280.27 | 5310.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 5329.50 | 5299.85 | 5298.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 5329.50 | 5299.85 | 5298.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 5375.50 | 5331.48 | 5317.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 5331.50 | 5338.53 | 5327.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 5315.00 | 5333.83 | 5326.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 5357.50 | 5333.83 | 5326.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 5345.50 | 5336.16 | 5328.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 5365.00 | 5341.05 | 5335.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 5314.50 | 5337.04 | 5334.89 | SL hit (close<static) qty=1.00 sl=5315.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 5313.00 | 5332.23 | 5332.90 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 5369.50 | 5334.58 | 5333.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 5377.00 | 5343.06 | 5337.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 5472.50 | 5440.03 | 5412.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 5477.00 | 5490.86 | 5472.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 5489.50 | 5491.79 | 5474.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 5471.50 | 5485.36 | 5474.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 5477.50 | 5483.79 | 5474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 5474.00 | 5483.79 | 5474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 5459.50 | 5478.93 | 5473.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 5472.00 | 5478.93 | 5473.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 5461.50 | 5475.45 | 5472.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 5526.00 | 5477.07 | 5474.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 5504.00 | 5515.28 | 5502.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 5510.00 | 5512.62 | 5502.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 5504.50 | 5509.40 | 5502.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 5495.50 | 5506.62 | 5501.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 5494.00 | 5506.62 | 5501.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 5499.50 | 5505.19 | 5501.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:30:00 | 5493.00 | 5505.19 | 5501.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 5486.00 | 5501.36 | 5500.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 5525.50 | 5501.36 | 5500.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 5520.00 | 5537.59 | 5539.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 5520.00 | 5537.59 | 5539.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 5500.50 | 5530.17 | 5536.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 5443.50 | 5429.52 | 5455.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 5434.50 | 5429.52 | 5455.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5432.50 | 5430.12 | 5453.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 5420.00 | 5430.00 | 5449.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 5409.00 | 5430.00 | 5449.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 5414.50 | 5419.14 | 5435.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 5461.00 | 5432.99 | 5437.00 | SL hit (close>static) qty=1.00 sl=5460.50 alert=retest2 |

### Cycle 173 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 5469.00 | 5442.91 | 5440.98 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 5401.50 | 5435.09 | 5439.53 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 5496.50 | 5449.42 | 5444.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 5501.00 | 5473.06 | 5458.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 5472.00 | 5472.85 | 5459.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 5472.00 | 5472.85 | 5459.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 5470.00 | 5483.45 | 5472.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 5473.50 | 5483.45 | 5472.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 5494.00 | 5485.56 | 5474.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:30:00 | 5497.00 | 5486.55 | 5476.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 5497.50 | 5486.44 | 5477.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 5499.00 | 5486.44 | 5477.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 5499.00 | 5490.96 | 5480.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 5488.50 | 5490.47 | 5481.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 5488.50 | 5490.47 | 5481.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 5487.00 | 5489.77 | 5482.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 5455.00 | 5478.77 | 5479.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 5455.00 | 5478.77 | 5479.63 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 5488.00 | 5474.94 | 5473.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 5509.00 | 5481.75 | 5476.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 5460.00 | 5478.00 | 5475.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 5483.00 | 5479.00 | 5476.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 5492.00 | 5481.58 | 5477.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 5492.50 | 5485.15 | 5480.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 5521.00 | 5554.66 | 5555.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 5521.00 | 5554.66 | 5555.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 5509.50 | 5540.89 | 5548.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 5543.00 | 5528.57 | 5538.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 5545.00 | 5531.86 | 5538.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 5554.00 | 5531.86 | 5538.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 5452.00 | 5431.84 | 5460.98 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 5527.00 | 5480.09 | 5475.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 5544.00 | 5492.87 | 5482.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 5459.50 | 5489.60 | 5481.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 5506.00 | 5492.88 | 5483.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 5530.50 | 5507.95 | 5494.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 5530.00 | 5502.43 | 5498.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:45:00 | 5529.00 | 5515.25 | 5505.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 5655.00 | 5721.64 | 5728.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 5655.00 | 5721.64 | 5728.13 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 5729.50 | 5707.38 | 5706.01 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 5669.00 | 5701.00 | 5703.56 | EMA200 below EMA400 |

### Cycle 183 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 5737.00 | 5707.82 | 5705.27 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 5686.00 | 5708.95 | 5709.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 5664.00 | 5699.96 | 5705.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 5691.50 | 5685.61 | 5695.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 5691.50 | 5685.61 | 5695.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 5701.50 | 5688.79 | 5695.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 5701.50 | 5688.79 | 5695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5709.00 | 5692.83 | 5697.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 5667.00 | 5692.83 | 5697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 5665.50 | 5687.36 | 5694.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 5636.00 | 5673.35 | 5686.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 5629.00 | 5657.24 | 5675.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 5723.50 | 5677.62 | 5671.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 5723.50 | 5677.62 | 5671.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 5732.50 | 5688.60 | 5677.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 5729.00 | 5733.01 | 5708.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 5729.00 | 5733.01 | 5708.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 5691.50 | 5724.71 | 5707.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 5691.50 | 5724.71 | 5707.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 5706.50 | 5721.07 | 5707.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 5716.50 | 5708.00 | 5703.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 14:15:00 | 5683.00 | 5699.41 | 5700.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 5683.00 | 5699.41 | 5700.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 5677.50 | 5695.03 | 5698.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 5631.00 | 5625.71 | 5647.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:45:00 | 5632.50 | 5625.71 | 5647.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 5625.00 | 5628.10 | 5644.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 5646.00 | 5628.10 | 5644.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 5629.00 | 5628.28 | 5643.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 5615.00 | 5625.62 | 5640.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 5680.00 | 5647.87 | 5646.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 5680.00 | 5647.87 | 5646.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 5709.50 | 5669.18 | 5660.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 5681.00 | 5684.01 | 5670.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:30:00 | 5680.00 | 5684.01 | 5670.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 5670.00 | 5681.21 | 5670.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 5660.00 | 5681.21 | 5670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 5685.00 | 5681.96 | 5671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 5681.50 | 5681.96 | 5671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 5675.50 | 5680.67 | 5671.95 | EMA400 retest candle locked (from upside) |

### Cycle 188 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 5611.00 | 5660.00 | 5664.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 5595.00 | 5637.40 | 5653.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 5634.00 | 5628.18 | 5645.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 5634.00 | 5628.18 | 5645.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 5659.00 | 5634.34 | 5646.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 5659.00 | 5634.34 | 5646.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 5656.50 | 5638.77 | 5647.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 5669.00 | 5638.77 | 5647.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5670.50 | 5645.12 | 5649.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 5666.50 | 5645.12 | 5649.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 189 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5672.00 | 5654.24 | 5653.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 5717.50 | 5666.89 | 5659.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 5663.50 | 5666.21 | 5659.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 5643.00 | 5661.57 | 5658.15 | EMA400 retest candle locked (from upside) |

### Cycle 190 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 5629.50 | 5655.16 | 5655.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 5611.00 | 5646.32 | 5651.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 5648.00 | 5622.74 | 5635.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5638.00 | 5625.79 | 5635.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 5660.50 | 5625.79 | 5635.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 5621.50 | 5624.94 | 5634.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 5597.00 | 5625.45 | 5633.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 5602.00 | 5620.32 | 5629.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 5608.50 | 5616.72 | 5625.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5650.00 | 5623.38 | 5627.89 | SL hit (close>static) qty=1.00 sl=5640.00 alert=retest2 |

### Cycle 191 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 5638.00 | 5622.40 | 5621.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 12:15:00 | 5654.00 | 5628.72 | 5624.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 5643.00 | 5649.14 | 5638.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 5639.00 | 5647.11 | 5638.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 5640.50 | 5647.11 | 5638.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 5608.00 | 5639.29 | 5635.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 5608.00 | 5639.29 | 5635.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 5600.50 | 5631.53 | 5632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 5557.50 | 5598.21 | 5614.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 5555.50 | 5540.46 | 5565.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 5582.50 | 5548.87 | 5566.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 5613.50 | 5548.87 | 5566.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 5593.50 | 5557.80 | 5569.16 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 5640.00 | 5580.27 | 5577.77 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 5554.00 | 5589.25 | 5592.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 5538.00 | 5573.52 | 5584.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 13:15:00 | 5572.50 | 5571.87 | 5581.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 5567.00 | 5571.87 | 5581.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 5514.50 | 5556.85 | 5572.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 5536.50 | 5556.85 | 5572.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5518.50 | 5534.77 | 5551.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:30:00 | 5508.50 | 5525.55 | 5539.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 5506.00 | 5474.02 | 5492.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 5510.00 | 5492.28 | 5496.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 5557.50 | 5488.80 | 5483.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 5557.50 | 5488.80 | 5483.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5597.50 | 5525.20 | 5502.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 5546.50 | 5567.98 | 5538.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 5563.00 | 5566.99 | 5540.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 5537.00 | 5566.99 | 5540.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 5569.50 | 5567.49 | 5543.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 5590.50 | 5567.49 | 5543.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 5580.00 | 5572.73 | 5550.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 5770.50 | 5823.62 | 5824.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 5770.50 | 5823.62 | 5824.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 5727.00 | 5804.29 | 5815.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 5780.00 | 5750.40 | 5775.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5722.00 | 5722.82 | 5749.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 5647.50 | 5706.46 | 5739.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 5744.50 | 5720.04 | 5716.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 5744.50 | 5720.04 | 5716.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 5794.50 | 5734.93 | 5723.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 5744.00 | 5756.45 | 5741.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5749.00 | 5754.96 | 5741.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 5767.50 | 5754.96 | 5741.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 5774.00 | 5758.77 | 5744.86 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 5700.00 | 5736.94 | 5741.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 5669.50 | 5723.45 | 5734.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 5724.00 | 5716.29 | 5729.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 5724.00 | 5716.29 | 5729.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 5720.00 | 5717.03 | 5728.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 5686.50 | 5717.03 | 5728.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5633.00 | 5700.23 | 5719.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 5625.00 | 5668.25 | 5684.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 5602.00 | 5655.00 | 5676.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 5703.00 | 5601.48 | 5598.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 5703.00 | 5601.48 | 5598.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 5735.00 | 5628.18 | 5611.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 5644.50 | 5664.59 | 5639.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 5674.00 | 5666.48 | 5642.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 5677.00 | 5666.48 | 5642.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 5660.00 | 5663.44 | 5648.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 5613.00 | 5663.44 | 5648.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5607.00 | 5652.15 | 5644.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 5594.50 | 5652.15 | 5644.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 5650.00 | 5651.72 | 5645.42 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5580.50 | 5640.70 | 5643.60 | EMA200 below EMA400 |

### Cycle 201 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 5674.00 | 5645.03 | 5644.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 5693.00 | 5654.62 | 5648.56 | Break + close above crossover candle high |

### Cycle 202 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 5587.50 | 5641.20 | 5643.01 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 5746.50 | 5662.26 | 5652.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 5771.00 | 5700.37 | 5672.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 5863.50 | 5865.86 | 5820.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:45:00 | 5855.50 | 5865.86 | 5820.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 5876.00 | 5859.20 | 5831.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 5846.00 | 5859.20 | 5831.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 5857.00 | 5858.76 | 5833.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 5798.00 | 5858.76 | 5833.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5773.00 | 5841.61 | 5827.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 5799.00 | 5841.61 | 5827.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 5770.50 | 5827.39 | 5822.69 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 5782.00 | 5818.31 | 5818.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 12:15:00 | 5735.00 | 5801.65 | 5811.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 5461.50 | 5446.36 | 5493.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 5461.50 | 5446.36 | 5493.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 5448.00 | 5451.40 | 5484.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 5408.50 | 5439.00 | 5472.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 5418.50 | 5390.10 | 5411.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 5490.00 | 5431.77 | 5425.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 5490.00 | 5431.77 | 5425.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 5500.50 | 5477.65 | 5459.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 5672.50 | 5690.08 | 5633.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 5673.00 | 5690.08 | 5633.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 5637.00 | 5673.04 | 5639.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 5637.00 | 5673.04 | 5639.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5614.00 | 5661.23 | 5636.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5572.00 | 5661.23 | 5636.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5560.00 | 5640.99 | 5629.79 | EMA400 retest candle locked (from upside) |

### Cycle 206 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 5540.50 | 5620.89 | 5621.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 5499.50 | 5596.61 | 5610.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 5582.50 | 5570.39 | 5590.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 5489.00 | 5570.39 | 5590.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5506.50 | 5494.45 | 5529.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 5522.50 | 5494.45 | 5529.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 5500.50 | 5491.21 | 5516.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 5500.50 | 5491.21 | 5516.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 5536.50 | 5500.27 | 5518.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 5536.50 | 5500.27 | 5518.44 | SL hit (close>ema400) qty=1.00 sl=5518.44 alert=retest1 |

### Cycle 207 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 5555.50 | 5525.67 | 5524.97 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 5461.00 | 5513.51 | 5519.72 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5620.00 | 5520.94 | 5512.71 | EMA200 above EMA400 |

### Cycle 210 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5467.50 | 5547.94 | 5550.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 5407.00 | 5486.23 | 5518.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 5322.00 | 5315.30 | 5361.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 5253.50 | 5358.86 | 5359.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 5377.00 | 5270.79 | 5248.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 5392.50 | 5394.65 | 5336.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:30:00 | 5382.00 | 5394.65 | 5336.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 5353.00 | 5375.01 | 5345.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 5353.00 | 5375.01 | 5345.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 5333.50 | 5366.70 | 5343.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 5333.50 | 5366.70 | 5343.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 5271.50 | 5347.66 | 5337.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 5268.00 | 5347.66 | 5337.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 5279.00 | 5333.93 | 5332.07 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 5301.00 | 5327.34 | 5329.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 5228.00 | 5285.50 | 5304.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 5250.00 | 5209.50 | 5247.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 5264.00 | 5220.40 | 5248.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 5264.00 | 5220.40 | 5248.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 5271.00 | 5230.52 | 5250.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 5221.50 | 5230.52 | 5250.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5211.00 | 5226.61 | 5247.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 5180.00 | 5217.29 | 5240.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 5174.50 | 5205.03 | 5233.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 5180.00 | 5200.03 | 5228.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 5157.50 | 5209.56 | 5224.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 5200.00 | 5207.65 | 5222.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:30:00 | 5217.50 | 5207.65 | 5222.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 5237.50 | 5194.05 | 5206.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 11:15:00 | 5254.50 | 5215.25 | 5214.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 5254.50 | 5215.25 | 5214.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 5336.50 | 5257.96 | 5238.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5408.50 | 5403.75 | 5355.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 5432.50 | 5395.23 | 5371.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 5539.50 | 5632.68 | 5640.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 214 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 5539.50 | 5632.68 | 5640.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 5443.00 | 5576.96 | 5613.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 15:15:00 | 5539.00 | 5538.60 | 5580.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 09:15:00 | 5559.50 | 5538.60 | 5580.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 5542.00 | 5539.28 | 5576.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 5503.50 | 5530.53 | 5569.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 5228.32 | 5441.76 | 5513.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 5419.50 | 5405.03 | 5482.09 | SL hit (close>ema200) qty=0.50 sl=5405.03 alert=retest2 |

### Cycle 215 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 5404.50 | 5378.54 | 5375.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 5509.50 | 5408.89 | 5389.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 5552.50 | 5569.90 | 5525.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 5628.00 | 5569.90 | 5525.80 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 5609.00 | 5577.72 | 5533.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 12:30:00 | 5610.50 | 5594.69 | 5553.65 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 14:00:00 | 5616.50 | 5599.05 | 5559.36 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 5560.00 | 5589.39 | 5561.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 5560.00 | 5589.39 | 5561.75 | SL hit (close<ema400) qty=1.00 sl=5561.75 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 09:15:00 | 3367.90 | 2023-05-23 09:15:00 | 3199.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-18 10:45:00 | 3362.85 | 2023-05-23 09:15:00 | 3194.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-05-18 09:15:00 | 3367.90 | 2023-05-23 15:15:00 | 3267.40 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2023-05-18 10:45:00 | 3362.85 | 2023-05-23 15:15:00 | 3267.40 | STOP_HIT | 0.50 | 2.84% |
| BUY | retest2 | 2023-05-30 10:30:00 | 3381.30 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-05-30 11:00:00 | 3380.00 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2023-05-31 10:15:00 | 3388.00 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2023-05-31 10:45:00 | 3381.70 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2023-06-01 09:15:00 | 3410.15 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2023-06-01 14:30:00 | 3379.05 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-06-02 11:00:00 | 3378.95 | 2023-06-02 12:15:00 | 3342.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2023-06-06 13:30:00 | 3385.60 | 2023-06-12 09:15:00 | 3385.45 | STOP_HIT | 1.00 | -0.00% |
| BUY | retest2 | 2023-07-04 09:15:00 | 3509.55 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2023-07-04 11:45:00 | 3504.45 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2023-07-04 13:30:00 | 3507.10 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2023-07-04 15:15:00 | 3504.00 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2023-07-05 12:15:00 | 3510.95 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-07-05 14:45:00 | 3517.95 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-07-07 14:15:00 | 3512.80 | 2023-07-07 14:15:00 | 3493.95 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-07-14 09:30:00 | 3523.55 | 2023-07-14 10:15:00 | 3508.85 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2023-07-21 10:45:00 | 3695.00 | 2023-08-02 09:15:00 | 3946.00 | STOP_HIT | 1.00 | 6.79% |
| BUY | retest2 | 2023-07-21 12:15:00 | 3693.95 | 2023-08-02 09:15:00 | 3946.00 | STOP_HIT | 1.00 | 6.82% |
| SELL | retest2 | 2023-08-24 11:00:00 | 3750.10 | 2023-09-06 09:15:00 | 3690.05 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2023-09-11 11:15:00 | 3638.05 | 2023-09-13 09:15:00 | 3674.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-09-12 09:30:00 | 3644.10 | 2023-09-13 09:15:00 | 3674.20 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2023-09-12 11:30:00 | 3642.95 | 2023-09-13 09:15:00 | 3674.20 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2023-09-22 09:15:00 | 3635.00 | 2023-09-29 10:15:00 | 3611.70 | STOP_HIT | 1.00 | 0.64% |
| BUY | retest2 | 2023-10-18 11:15:00 | 3647.10 | 2023-10-20 09:15:00 | 3590.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2023-10-23 11:30:00 | 3566.45 | 2023-10-27 09:15:00 | 3624.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2023-10-23 14:30:00 | 3562.35 | 2023-10-27 09:15:00 | 3624.25 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2023-10-25 11:15:00 | 3565.95 | 2023-10-27 09:15:00 | 3624.25 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2023-10-26 09:15:00 | 3552.25 | 2023-10-27 09:15:00 | 3624.25 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2023-12-04 13:45:00 | 4644.70 | 2023-12-04 15:15:00 | 4615.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2023-12-08 15:15:00 | 4799.00 | 2023-12-11 09:15:00 | 4729.90 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-01-02 09:15:00 | 5161.40 | 2024-01-08 10:15:00 | 5174.95 | STOP_HIT | 1.00 | 0.26% |
| SELL | retest2 | 2024-01-17 09:15:00 | 5048.65 | 2024-01-17 09:15:00 | 5068.70 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-01-20 10:30:00 | 4920.65 | 2024-01-24 12:15:00 | 4939.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2024-01-23 09:45:00 | 4906.00 | 2024-01-24 12:15:00 | 4939.55 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-01-24 09:45:00 | 4918.65 | 2024-01-24 12:15:00 | 4939.55 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-01-24 12:15:00 | 4918.50 | 2024-01-24 12:15:00 | 4939.55 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2024-02-01 09:15:00 | 5033.05 | 2024-02-01 10:15:00 | 4970.15 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-02-20 12:15:00 | 5400.30 | 2024-02-23 10:15:00 | 5382.80 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-02-20 15:00:00 | 5407.95 | 2024-02-23 10:15:00 | 5382.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-02-22 14:15:00 | 5396.90 | 2024-02-23 10:15:00 | 5382.80 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2024-02-22 14:45:00 | 5414.00 | 2024-02-23 10:15:00 | 5382.80 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-03-11 09:15:00 | 5178.55 | 2024-03-12 09:15:00 | 5132.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-03-15 10:15:00 | 4902.90 | 2024-03-26 09:15:00 | 4657.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-15 10:15:00 | 4902.90 | 2024-03-26 09:15:00 | 4792.90 | STOP_HIT | 0.50 | 2.24% |
| SELL | retest2 | 2024-03-19 10:15:00 | 4902.00 | 2024-03-26 09:15:00 | 4656.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-19 10:15:00 | 4902.00 | 2024-03-26 09:15:00 | 4792.90 | STOP_HIT | 0.50 | 2.23% |
| BUY | retest2 | 2024-03-28 12:15:00 | 4946.00 | 2024-04-03 09:15:00 | 4925.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2024-03-28 15:15:00 | 4949.80 | 2024-04-03 09:15:00 | 4925.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-04-02 10:00:00 | 4989.35 | 2024-04-03 09:15:00 | 4925.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-04-02 14:30:00 | 4940.55 | 2024-04-03 09:15:00 | 4925.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-04-09 12:15:00 | 4870.20 | 2024-04-19 09:15:00 | 4626.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 12:45:00 | 4875.45 | 2024-04-19 09:15:00 | 4631.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 15:15:00 | 4867.50 | 2024-04-19 09:15:00 | 4624.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 12:15:00 | 4870.20 | 2024-04-22 09:15:00 | 4648.85 | STOP_HIT | 0.50 | 4.54% |
| SELL | retest2 | 2024-04-09 12:45:00 | 4875.45 | 2024-04-22 09:15:00 | 4648.85 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2024-04-09 15:15:00 | 4867.50 | 2024-04-22 09:15:00 | 4648.85 | STOP_HIT | 0.50 | 4.49% |
| BUY | retest2 | 2024-04-30 09:15:00 | 4937.40 | 2024-04-30 12:15:00 | 4864.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-10 10:30:00 | 5131.05 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 2.08% |
| BUY | retest2 | 2024-05-10 11:30:00 | 5151.00 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-05-13 09:15:00 | 5179.60 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-05-13 10:15:00 | 5127.00 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2024-05-17 10:45:00 | 5329.95 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-05-21 11:00:00 | 5363.00 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-05-21 12:30:00 | 5337.45 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-21 15:00:00 | 5330.20 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-05-24 11:45:00 | 5472.80 | 2024-05-27 12:15:00 | 5325.50 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-05-24 13:00:00 | 5470.00 | 2024-05-27 12:15:00 | 5325.50 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-06-20 11:45:00 | 5186.50 | 2024-06-21 11:15:00 | 5127.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-26 09:15:00 | 5006.15 | 2024-07-03 11:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-06-26 13:00:00 | 5027.15 | 2024-07-03 11:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-07-10 12:30:00 | 5225.00 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-10 13:45:00 | 5220.20 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-07-10 14:15:00 | 5221.55 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-07-11 14:30:00 | 5231.15 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-12 09:15:00 | 5252.95 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-07-12 10:45:00 | 5212.85 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-23 12:15:00 | 5111.55 | 2024-07-24 09:15:00 | 5231.15 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-07-23 13:30:00 | 5157.65 | 2024-07-24 09:15:00 | 5231.15 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-30 15:15:00 | 5205.00 | 2024-07-31 09:15:00 | 5290.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-08-02 13:45:00 | 5324.80 | 2024-08-05 10:15:00 | 5255.90 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-08-02 14:15:00 | 5323.65 | 2024-08-05 10:15:00 | 5255.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-08-05 13:30:00 | 5339.95 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 5.82% |
| BUY | retest2 | 2024-08-06 09:15:00 | 5350.00 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2024-08-07 09:15:00 | 5442.05 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2024-08-21 09:30:00 | 5761.90 | 2024-09-05 10:15:00 | 6338.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 09:30:00 | 5776.30 | 2024-09-10 09:15:00 | 6353.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-09-19 12:45:00 | 6115.70 | 2024-09-24 10:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2024-09-20 14:30:00 | 6055.70 | 2024-09-25 11:15:00 | 6110.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-23 10:00:00 | 6093.35 | 2024-09-25 11:15:00 | 6110.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-10-25 13:45:00 | 5908.95 | 2024-10-28 10:15:00 | 6016.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-30 10:30:00 | 5886.60 | 2024-11-06 13:15:00 | 5828.55 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-11-14 10:00:00 | 5512.65 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-11-14 11:45:00 | 5516.90 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-11-14 14:45:00 | 5521.95 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-18 10:15:00 | 5519.50 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-11-21 14:15:00 | 5586.35 | 2024-11-22 10:15:00 | 5529.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-12-12 09:15:00 | 5449.30 | 2024-12-18 09:15:00 | 5502.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-12-19 10:30:00 | 5502.45 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-12-19 15:00:00 | 5501.95 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-12-20 09:30:00 | 5504.60 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-20 13:15:00 | 5500.50 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-23 15:15:00 | 5166.05 | 2025-01-27 10:15:00 | 4907.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 5166.05 | 2025-01-29 09:15:00 | 4930.80 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-02-04 09:15:00 | 5073.45 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2025-02-04 09:45:00 | 5061.55 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2025-02-04 10:15:00 | 5066.50 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-02-14 09:15:00 | 4624.95 | 2025-02-17 13:15:00 | 4754.10 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-02-17 10:00:00 | 4689.55 | 2025-02-17 13:15:00 | 4754.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-02-28 09:45:00 | 4510.20 | 2025-03-03 13:15:00 | 4607.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-03-05 12:00:00 | 4649.15 | 2025-03-10 14:15:00 | 4624.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-03-05 14:45:00 | 4684.70 | 2025-03-10 14:15:00 | 4624.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-03-13 10:30:00 | 4763.00 | 2025-03-13 14:15:00 | 4704.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-01 12:45:00 | 4841.05 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-04-01 15:00:00 | 4830.55 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-04-02 09:45:00 | 4830.00 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-04-02 11:00:00 | 4847.20 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-04-15 14:00:00 | 4893.60 | 2025-04-25 12:15:00 | 5041.20 | STOP_HIT | 1.00 | 3.02% |
| SELL | retest2 | 2025-05-12 11:15:00 | 5006.00 | 2025-05-12 11:15:00 | 5019.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-19 10:30:00 | 5391.50 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-05-19 13:00:00 | 5388.50 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-21 09:30:00 | 5377.00 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-21 10:00:00 | 5379.00 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-06-03 11:30:00 | 4973.50 | 2025-06-10 10:15:00 | 4942.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-06-03 15:15:00 | 4980.00 | 2025-06-10 10:15:00 | 4942.00 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-06-19 10:30:00 | 4763.00 | 2025-06-24 09:15:00 | 4800.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-20 12:15:00 | 4761.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-20 13:15:00 | 4763.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-20 14:15:00 | 4765.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-23 15:15:00 | 4757.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-07-04 11:30:00 | 4852.90 | 2025-07-04 12:15:00 | 4884.10 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-04 12:15:00 | 4858.70 | 2025-07-04 12:15:00 | 4884.10 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-07 12:15:00 | 4865.20 | 2025-07-08 10:15:00 | 4813.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-28 10:15:00 | 5071.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-29 09:30:00 | 5055.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-07-29 11:30:00 | 5055.90 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-07-30 09:30:00 | 5058.10 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-31 11:30:00 | 5038.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-08-06 09:30:00 | 4841.50 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-08-06 10:30:00 | 4850.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-06 13:45:00 | 4846.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-06 15:15:00 | 4840.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-07 12:00:00 | 4849.50 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-08-08 09:15:00 | 4844.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-08-11 14:45:00 | 4849.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-21 10:30:00 | 5439.00 | 2025-08-26 10:15:00 | 5405.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-21 15:15:00 | 5443.00 | 2025-08-26 10:15:00 | 5405.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-02 11:45:00 | 5245.50 | 2025-09-04 10:15:00 | 5329.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-09-02 13:00:00 | 5240.50 | 2025-09-04 10:15:00 | 5329.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-09-09 09:15:00 | 5357.50 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-09 10:00:00 | 5345.50 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-10 09:15:00 | 5365.00 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-15 12:00:00 | 5472.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-09-17 11:15:00 | 5477.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-09-17 11:45:00 | 5489.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-09-17 14:00:00 | 5471.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-09-19 09:15:00 | 5526.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-22 10:45:00 | 5504.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-09-22 11:30:00 | 5510.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-09-22 12:45:00 | 5504.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-09-23 09:15:00 | 5525.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-30 11:30:00 | 5420.00 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-30 12:15:00 | 5409.00 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-01 10:00:00 | 5414.50 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-08 13:30:00 | 5497.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-08 14:30:00 | 5497.50 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-08 15:15:00 | 5499.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-09 10:00:00 | 5499.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-14 12:45:00 | 5492.00 | 2025-10-23 14:15:00 | 5521.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-10-14 14:30:00 | 5492.50 | 2025-10-23 14:15:00 | 5521.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-10-30 15:00:00 | 5530.50 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2025-11-03 09:15:00 | 5530.00 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2025-11-03 11:45:00 | 5529.00 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2025-11-24 12:00:00 | 5636.00 | 2025-11-26 09:15:00 | 5723.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-24 15:00:00 | 5629.00 | 2025-11-26 09:15:00 | 5723.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-11-28 10:45:00 | 5716.50 | 2025-11-28 14:15:00 | 5683.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-03 10:45:00 | 5615.00 | 2025-12-04 09:15:00 | 5680.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-11 14:30:00 | 5597.00 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-12 10:45:00 | 5602.00 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-12 12:30:00 | 5608.50 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-12 15:15:00 | 5614.50 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-12-15 09:15:00 | 5587.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-15 11:15:00 | 5607.50 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-15 14:30:00 | 5607.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-15 15:15:00 | 5611.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-29 14:30:00 | 5508.50 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-31 10:45:00 | 5506.00 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-31 15:00:00 | 5510.00 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 09:15:00 | 5590.50 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2026-01-06 10:30:00 | 5580.00 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2026-01-21 10:30:00 | 5647.50 | 2026-01-22 15:15:00 | 5744.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-01 10:30:00 | 5625.00 | 2026-02-03 11:15:00 | 5703.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-01 12:00:00 | 5602.00 | 2026-02-03 11:15:00 | 5703.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-19 12:00:00 | 5408.50 | 2026-02-23 13:15:00 | 5490.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-02-23 09:45:00 | 5418.50 | 2026-02-23 13:15:00 | 5490.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest1 | 2026-03-04 09:15:00 | 5489.00 | 2026-03-05 14:15:00 | 5536.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-03-19 09:15:00 | 5253.50 | 2026-03-24 14:15:00 | 5294.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-04-06 11:00:00 | 5180.00 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-06 11:30:00 | 5174.50 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-06 13:00:00 | 5180.00 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-07 09:30:00 | 5157.50 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-04-13 10:15:00 | 5408.50 | 2026-04-23 09:15:00 | 5539.50 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2026-04-15 09:30:00 | 5432.50 | 2026-04-23 09:15:00 | 5539.50 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2026-04-24 10:30:00 | 5503.50 | 2026-04-24 14:15:00 | 5228.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 10:30:00 | 5503.50 | 2026-04-27 09:15:00 | 5419.50 | STOP_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2026-05-08 09:15:00 | 5628.00 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-05-08 10:00:00 | 5609.00 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest1 | 2026-05-08 12:30:00 | 5610.50 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2026-05-08 14:00:00 | 5616.50 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -1.01% |
