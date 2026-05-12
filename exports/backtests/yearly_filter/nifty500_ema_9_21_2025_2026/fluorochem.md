# Gujarat Fluorochemicals Ltd. (FLUOROCHEM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 3777.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 75 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 34 |
| ALERT3 | 144 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 72 |
| PARTIAL | 11 |
| TARGET_HIT | 0 |
| STOP_HIT | 75 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 86 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 39
- **Target hits / Stop hits / Partials:** 0 / 75 / 11
- **Avg / median % per leg:** 0.80% / 0.14%
- **Sum % (uncompounded):** 68.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 40 | 16 | 40.0% | 0 | 40 | 0 | -0.01% | -0.2% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.81% | -2.4% |
| BUY @ 3rd Alert (retest2) | 37 | 16 | 43.2% | 0 | 37 | 0 | 0.06% | 2.2% |
| SELL (all) | 46 | 31 | 67.4% | 0 | 35 | 11 | 1.50% | 69.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 46 | 31 | 67.4% | 0 | 35 | 11 | 1.50% | 69.1% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.81% | -2.4% |
| retest2 (combined) | 83 | 47 | 56.6% | 0 | 72 | 11 | 0.86% | 71.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 13:15:00 | 3873.20 | 3861.97 | 3861.26 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 11:15:00 | 3843.60 | 3860.55 | 3861.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 12:15:00 | 3829.40 | 3854.32 | 3858.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 3851.10 | 3848.63 | 3854.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 14:15:00 | 3851.10 | 3848.63 | 3854.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 3851.10 | 3848.63 | 3854.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 3851.10 | 3848.63 | 3854.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 09:15:00 | 3898.70 | 3858.71 | 3858.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 10:15:00 | 3945.70 | 3876.10 | 3866.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 3949.00 | 3959.61 | 3938.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 3949.00 | 3959.61 | 3938.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 3949.00 | 3959.61 | 3938.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 3955.00 | 3959.61 | 3938.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 3989.30 | 3963.74 | 3945.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 15:00:00 | 3989.30 | 3963.74 | 3945.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 3978.90 | 3970.33 | 3951.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:15:00 | 3983.80 | 3970.33 | 3951.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:30:00 | 4001.60 | 3970.48 | 3958.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:30:00 | 3990.30 | 3977.53 | 3963.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 11:00:00 | 3993.90 | 3980.80 | 3966.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 4021.00 | 4008.89 | 3989.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 14:45:00 | 4033.00 | 4017.58 | 4001.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 4037.00 | 4021.25 | 4012.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 15:15:00 | 3949.00 | 4001.67 | 4006.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 3949.00 | 4001.67 | 4006.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 3784.10 | 3958.15 | 3985.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 09:15:00 | 3645.80 | 3597.64 | 3653.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:00:00 | 3645.80 | 3597.64 | 3653.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 3653.20 | 3619.80 | 3650.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:30:00 | 3681.60 | 3619.80 | 3650.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 3610.20 | 3617.88 | 3647.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:30:00 | 3659.40 | 3617.88 | 3647.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3679.60 | 3633.57 | 3647.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:00:00 | 3679.60 | 3633.57 | 3647.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 3698.20 | 3646.50 | 3651.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 10:30:00 | 3703.10 | 3646.50 | 3651.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 13:15:00 | 3667.20 | 3656.64 | 3655.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 14:15:00 | 3677.30 | 3660.77 | 3657.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 3657.00 | 3667.94 | 3662.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 3657.00 | 3667.94 | 3662.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 3657.00 | 3667.94 | 3662.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 11:30:00 | 3658.90 | 3667.94 | 3662.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 3651.00 | 3664.55 | 3661.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:30:00 | 3650.00 | 3664.55 | 3661.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 3666.00 | 3663.04 | 3661.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 3654.50 | 3663.04 | 3661.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 3647.60 | 3659.95 | 3660.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 3637.60 | 3652.10 | 3656.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 15:15:00 | 3655.00 | 3652.68 | 3656.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 15:15:00 | 3655.00 | 3652.68 | 3656.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 3655.00 | 3652.68 | 3656.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 3653.60 | 3652.68 | 3656.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 3671.50 | 3656.45 | 3657.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:30:00 | 3682.00 | 3656.45 | 3657.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 3661.10 | 3657.38 | 3657.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 3676.90 | 3657.38 | 3657.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 11:15:00 | 3661.00 | 3658.10 | 3658.10 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 3630.00 | 3652.48 | 3655.54 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 3672.00 | 3655.41 | 3654.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 15:15:00 | 3702.00 | 3666.89 | 3660.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 3665.00 | 3666.51 | 3660.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 3665.00 | 3666.51 | 3660.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 3665.00 | 3666.51 | 3660.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 3665.00 | 3666.51 | 3660.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3648.20 | 3662.85 | 3659.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 3648.20 | 3662.85 | 3659.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3637.30 | 3657.74 | 3657.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 3637.30 | 3657.74 | 3657.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 12:15:00 | 3643.70 | 3654.93 | 3656.19 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 3769.90 | 3677.10 | 3666.00 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 3665.20 | 3672.97 | 3673.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 3626.90 | 3661.06 | 3667.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 3729.60 | 3669.14 | 3669.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 3729.60 | 3669.14 | 3669.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 3729.60 | 3669.14 | 3669.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 3729.60 | 3669.14 | 3669.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 3736.00 | 3682.51 | 3675.74 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 11:15:00 | 3659.40 | 3671.04 | 3671.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 12:15:00 | 3628.30 | 3662.49 | 3667.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 09:15:00 | 3649.20 | 3647.63 | 3657.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 3649.20 | 3647.63 | 3657.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 3649.20 | 3647.63 | 3657.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 3659.80 | 3647.63 | 3657.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3400.90 | 3386.23 | 3421.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 3400.90 | 3386.23 | 3421.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 3454.00 | 3399.78 | 3424.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 3454.00 | 3399.78 | 3424.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 3510.00 | 3421.82 | 3431.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:45:00 | 3514.50 | 3421.82 | 3431.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 3455.70 | 3438.20 | 3438.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 3515.00 | 3459.95 | 3448.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 3438.00 | 3462.83 | 3452.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 11:15:00 | 3438.00 | 3462.83 | 3452.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 3438.00 | 3462.83 | 3452.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:00:00 | 3438.00 | 3462.83 | 3452.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 3447.60 | 3459.79 | 3451.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 3447.00 | 3459.79 | 3451.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 3438.30 | 3455.49 | 3450.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 3435.00 | 3455.49 | 3450.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 3461.00 | 3458.08 | 3452.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 09:15:00 | 3474.90 | 3458.08 | 3452.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 3525.00 | 3557.20 | 3561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 3525.00 | 3557.20 | 3561.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 09:15:00 | 3508.70 | 3527.76 | 3538.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 3528.80 | 3521.01 | 3530.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 3528.80 | 3521.01 | 3530.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3528.80 | 3521.01 | 3530.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 14:45:00 | 3531.00 | 3521.01 | 3530.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 3513.00 | 3519.41 | 3528.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:15:00 | 3494.40 | 3519.41 | 3528.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 3505.00 | 3515.59 | 3526.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 3510.00 | 3514.88 | 3524.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 3503.00 | 3509.22 | 3518.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 3468.50 | 3500.08 | 3512.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 3461.30 | 3491.30 | 3507.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 3437.20 | 3420.15 | 3419.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 3437.20 | 3420.15 | 3419.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 3452.10 | 3426.54 | 3422.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 3440.00 | 3441.21 | 3433.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 11:15:00 | 3440.00 | 3441.21 | 3433.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 3440.00 | 3441.21 | 3433.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 3439.50 | 3441.21 | 3433.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 3430.00 | 3438.97 | 3432.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 3430.00 | 3438.97 | 3432.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 3436.90 | 3438.55 | 3433.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 3430.00 | 3438.55 | 3433.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 3424.20 | 3435.68 | 3432.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 3424.20 | 3435.68 | 3432.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 3425.00 | 3433.55 | 3431.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 3423.30 | 3431.84 | 3431.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 10:15:00 | 3425.40 | 3430.55 | 3430.64 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 12:15:00 | 3435.50 | 3430.70 | 3430.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 09:15:00 | 3490.00 | 3444.45 | 3437.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 3552.30 | 3565.95 | 3535.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 3552.30 | 3565.95 | 3535.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 3552.30 | 3565.95 | 3535.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 3552.30 | 3565.95 | 3535.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 3553.40 | 3575.52 | 3561.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 3554.10 | 3575.52 | 3561.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 3565.20 | 3573.46 | 3562.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 13:45:00 | 3575.00 | 3572.81 | 3563.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:45:00 | 3585.20 | 3573.89 | 3565.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 3526.30 | 3562.15 | 3561.07 | SL hit (close<static) qty=1.00 sl=3540.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 3509.80 | 3551.68 | 3556.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 3502.70 | 3522.97 | 3537.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 3490.60 | 3483.26 | 3503.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 13:00:00 | 3490.60 | 3483.26 | 3503.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 3513.80 | 3489.37 | 3504.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 3506.70 | 3489.37 | 3504.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 3529.50 | 3497.40 | 3506.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 3529.90 | 3497.40 | 3506.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 3557.50 | 3516.23 | 3514.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 14:15:00 | 3596.50 | 3558.37 | 3537.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 10:15:00 | 3540.30 | 3560.73 | 3544.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 10:15:00 | 3540.30 | 3560.73 | 3544.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 3540.30 | 3560.73 | 3544.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 3540.30 | 3560.73 | 3544.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 3582.90 | 3565.16 | 3548.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:15:00 | 3588.80 | 3565.16 | 3548.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 11:15:00 | 3531.00 | 3563.40 | 3558.40 | SL hit (close<static) qty=1.00 sl=3540.30 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 3524.10 | 3550.87 | 3553.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 3488.00 | 3538.29 | 3547.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 3516.40 | 3511.77 | 3525.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 3516.40 | 3511.77 | 3525.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3516.40 | 3511.77 | 3525.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 3528.20 | 3511.77 | 3525.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 3521.00 | 3513.61 | 3525.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 3536.90 | 3513.61 | 3525.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 3537.10 | 3518.31 | 3526.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 3443.60 | 3504.79 | 3517.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 09:15:00 | 3646.90 | 3528.61 | 3524.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 3646.90 | 3528.61 | 3524.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 10:15:00 | 3662.40 | 3555.36 | 3537.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 3583.40 | 3613.56 | 3582.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 3583.40 | 3613.56 | 3582.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3583.40 | 3613.56 | 3582.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:15:00 | 3596.00 | 3613.56 | 3582.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 3565.80 | 3604.01 | 3580.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 3565.80 | 3604.01 | 3580.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 3538.70 | 3590.95 | 3577.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:00:00 | 3538.70 | 3590.95 | 3577.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 3590.00 | 3582.43 | 3575.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 3563.20 | 3582.43 | 3575.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3542.00 | 3574.34 | 3572.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 3544.00 | 3574.34 | 3572.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 10:15:00 | 3526.50 | 3564.77 | 3568.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 11:15:00 | 3499.40 | 3551.70 | 3562.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-12 14:15:00 | 3439.70 | 3422.27 | 3453.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-12 15:00:00 | 3439.70 | 3422.27 | 3453.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 3439.40 | 3426.47 | 3449.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:45:00 | 3423.80 | 3426.12 | 3447.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:45:00 | 3426.00 | 3425.72 | 3443.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 3421.40 | 3425.72 | 3443.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 3420.00 | 3404.98 | 3411.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 3392.50 | 3402.91 | 3409.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 14:00:00 | 3388.80 | 3401.36 | 3406.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 15:00:00 | 3383.50 | 3397.79 | 3404.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 14:15:00 | 3419.10 | 3407.36 | 3406.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 14:15:00 | 3419.10 | 3407.36 | 3406.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 3433.80 | 3413.89 | 3409.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 3415.00 | 3419.57 | 3413.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:00:00 | 3415.00 | 3419.57 | 3413.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 3400.00 | 3415.65 | 3412.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:45:00 | 3399.90 | 3415.65 | 3412.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 3389.00 | 3410.32 | 3410.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 3389.00 | 3410.32 | 3410.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 3385.00 | 3405.26 | 3407.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 3375.50 | 3399.31 | 3404.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 13:15:00 | 3401.00 | 3387.39 | 3396.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 13:15:00 | 3401.00 | 3387.39 | 3396.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 3401.00 | 3387.39 | 3396.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:00:00 | 3401.00 | 3387.39 | 3396.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3417.00 | 3393.31 | 3398.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:30:00 | 3425.00 | 3393.31 | 3398.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 3415.00 | 3397.65 | 3399.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 3413.50 | 3397.65 | 3399.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 3396.80 | 3399.89 | 3400.35 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 13:15:00 | 3405.30 | 3400.97 | 3400.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 14:15:00 | 3449.00 | 3410.58 | 3405.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 3402.00 | 3415.17 | 3408.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 3402.00 | 3415.17 | 3408.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3402.00 | 3415.17 | 3408.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 3385.30 | 3415.17 | 3408.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3395.50 | 3411.24 | 3407.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 3396.30 | 3411.24 | 3407.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 12:15:00 | 3393.80 | 3404.93 | 3405.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 3373.30 | 3396.53 | 3401.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 3395.60 | 3392.90 | 3398.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 3395.60 | 3392.90 | 3398.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 3395.60 | 3392.90 | 3398.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 3395.60 | 3392.90 | 3398.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 3394.90 | 3393.30 | 3398.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 3400.00 | 3393.30 | 3398.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 3420.00 | 3398.64 | 3400.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:30:00 | 3423.00 | 3398.64 | 3400.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 12:15:00 | 3425.00 | 3403.91 | 3402.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 13:15:00 | 3430.00 | 3409.13 | 3404.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 09:15:00 | 3403.40 | 3410.45 | 3406.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 09:15:00 | 3403.40 | 3410.45 | 3406.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3403.40 | 3410.45 | 3406.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 09:30:00 | 3406.80 | 3410.45 | 3406.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 3424.90 | 3413.34 | 3408.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 11:15:00 | 3441.70 | 3413.34 | 3408.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:45:00 | 3428.90 | 3416.56 | 3411.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 15:15:00 | 3398.60 | 3412.97 | 3410.63 | SL hit (close<static) qty=1.00 sl=3401.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-01 12:15:00 | 3403.70 | 3409.26 | 3409.61 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 3438.90 | 3410.07 | 3409.27 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 3401.50 | 3411.07 | 3411.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 14:15:00 | 3380.50 | 3398.01 | 3404.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 3395.40 | 3393.82 | 3401.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 3395.40 | 3393.82 | 3401.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3395.40 | 3393.82 | 3401.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:45:00 | 3410.00 | 3393.82 | 3401.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 3401.00 | 3395.26 | 3401.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:00:00 | 3392.00 | 3394.61 | 3400.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:15:00 | 3391.90 | 3390.53 | 3395.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 3389.60 | 3390.02 | 3395.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 3440.00 | 3385.03 | 3388.00 | SL hit (close>static) qty=1.00 sl=3404.80 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 3414.90 | 3391.01 | 3390.44 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 14:15:00 | 3360.00 | 3387.89 | 3389.78 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 3449.00 | 3401.92 | 3395.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 3547.10 | 3430.96 | 3409.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 3628.80 | 3676.10 | 3606.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 3628.80 | 3676.10 | 3606.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 3628.80 | 3676.10 | 3606.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 3617.40 | 3676.10 | 3606.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 3617.70 | 3664.42 | 3607.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 3671.00 | 3627.69 | 3607.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 09:15:00 | 3616.00 | 3670.49 | 3672.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 09:15:00 | 3616.00 | 3670.49 | 3672.96 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 3697.00 | 3675.31 | 3672.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 10:15:00 | 3727.70 | 3687.10 | 3678.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 09:15:00 | 3855.00 | 3865.23 | 3821.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:00:00 | 3855.00 | 3865.23 | 3821.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 3835.00 | 3871.47 | 3840.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 13:45:00 | 3830.20 | 3871.47 | 3840.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 3846.60 | 3866.49 | 3840.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 3865.00 | 3866.49 | 3840.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 3812.90 | 3846.02 | 3839.83 | SL hit (close<static) qty=1.00 sl=3825.70 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 14:15:00 | 3789.20 | 3827.29 | 3831.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 15:15:00 | 3785.00 | 3818.84 | 3827.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 14:15:00 | 3778.00 | 3768.67 | 3793.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 14:45:00 | 3783.80 | 3768.67 | 3793.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 3780.00 | 3770.94 | 3792.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 3736.60 | 3770.94 | 3792.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 3738.60 | 3705.06 | 3702.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 3738.60 | 3705.06 | 3702.79 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 3683.10 | 3700.03 | 3701.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 12:15:00 | 3617.40 | 3674.60 | 3689.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 3685.30 | 3660.84 | 3676.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 3685.30 | 3660.84 | 3676.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 3685.30 | 3660.84 | 3676.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:00:00 | 3685.30 | 3660.84 | 3676.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 3685.40 | 3665.75 | 3676.97 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 15:15:00 | 3691.80 | 3682.97 | 3682.33 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 3679.90 | 3682.66 | 3682.73 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 14:15:00 | 3725.10 | 3691.14 | 3686.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 3792.10 | 3713.89 | 3699.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 12:15:00 | 3733.00 | 3734.52 | 3717.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 15:15:00 | 3725.00 | 3731.08 | 3719.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 3725.00 | 3731.08 | 3719.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 3693.70 | 3731.08 | 3719.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 3692.00 | 3723.26 | 3717.40 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3671.00 | 3706.25 | 3710.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 3665.00 | 3698.00 | 3706.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 09:15:00 | 3721.70 | 3651.58 | 3658.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 09:15:00 | 3721.70 | 3651.58 | 3658.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 3721.70 | 3651.58 | 3658.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:00:00 | 3721.70 | 3651.58 | 3658.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 3711.10 | 3663.48 | 3662.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 3749.40 | 3699.36 | 3682.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 13:15:00 | 3700.00 | 3718.50 | 3702.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 3700.00 | 3718.50 | 3702.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3700.00 | 3718.50 | 3702.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 3700.00 | 3718.50 | 3702.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 3731.30 | 3721.06 | 3704.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 15:15:00 | 3692.50 | 3721.06 | 3704.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 3692.50 | 3715.35 | 3703.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 3734.20 | 3715.35 | 3703.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:30:00 | 3760.00 | 3719.98 | 3708.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:30:00 | 3732.00 | 3722.54 | 3710.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 13:00:00 | 3732.80 | 3722.54 | 3710.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 3720.90 | 3730.22 | 3721.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 3720.90 | 3730.22 | 3721.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 3709.70 | 3726.11 | 3720.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 3715.50 | 3726.11 | 3720.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 3737.80 | 3728.45 | 3722.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 3728.90 | 3728.45 | 3722.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 3732.10 | 3729.41 | 3723.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 3732.10 | 3729.41 | 3723.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 3731.30 | 3729.79 | 3724.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 3725.50 | 3729.79 | 3724.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 3722.70 | 3728.37 | 3724.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 3722.70 | 3728.37 | 3724.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 3710.70 | 3724.84 | 3723.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:30:00 | 3713.20 | 3724.84 | 3723.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-24 12:15:00 | 3689.90 | 3716.32 | 3719.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-10-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 12:15:00 | 3689.90 | 3716.32 | 3719.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 3679.40 | 3708.94 | 3715.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3709.80 | 3709.11 | 3715.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 3709.80 | 3709.11 | 3715.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3709.80 | 3709.11 | 3715.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:30:00 | 3729.40 | 3709.11 | 3715.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3694.90 | 3706.27 | 3713.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 3683.60 | 3706.27 | 3713.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 09:15:00 | 3718.00 | 3643.09 | 3634.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 3718.00 | 3643.09 | 3634.68 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 3655.50 | 3695.62 | 3697.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 3654.70 | 3682.90 | 3691.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 3575.30 | 3559.13 | 3596.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 15:00:00 | 3575.30 | 3559.13 | 3596.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 3634.20 | 3531.39 | 3552.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 3634.20 | 3531.39 | 3552.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 3635.40 | 3552.19 | 3560.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 3635.40 | 3552.19 | 3560.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 3665.00 | 3574.75 | 3569.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 3670.90 | 3607.64 | 3586.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 3630.40 | 3636.73 | 3607.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 09:30:00 | 3635.40 | 3636.73 | 3607.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 3609.00 | 3628.91 | 3608.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:00:00 | 3609.00 | 3628.91 | 3608.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 3618.50 | 3626.83 | 3609.62 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 3546.30 | 3593.70 | 3599.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 3512.50 | 3568.77 | 3578.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 14:15:00 | 3573.70 | 3536.94 | 3555.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 14:15:00 | 3573.70 | 3536.94 | 3555.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 3573.70 | 3536.94 | 3555.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 3573.70 | 3536.94 | 3555.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 3582.30 | 3546.01 | 3557.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 3553.10 | 3546.01 | 3557.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 13:15:00 | 3375.44 | 3420.11 | 3454.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 15:15:00 | 3548.00 | 3436.52 | 3455.03 | SL hit (close>ema200) qty=0.50 sl=3436.52 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 3467.30 | 3441.12 | 3438.14 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 3421.10 | 3435.05 | 3436.42 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-11-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 14:15:00 | 3470.00 | 3442.04 | 3439.47 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 3430.50 | 3439.74 | 3440.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 3402.90 | 3430.81 | 3436.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 3431.00 | 3404.85 | 3416.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 3431.00 | 3404.85 | 3416.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 3431.00 | 3404.85 | 3416.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 3431.00 | 3404.85 | 3416.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 3410.00 | 3405.88 | 3415.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 11:15:00 | 3407.30 | 3405.88 | 3415.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 3398.10 | 3414.07 | 3416.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 3395.50 | 3363.40 | 3368.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 3236.93 | 3299.68 | 3332.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 3228.19 | 3299.68 | 3332.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 3225.72 | 3299.68 | 3332.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 14:15:00 | 3320.00 | 3303.74 | 3331.67 | SL hit (close>ema200) qty=0.50 sl=3303.74 alert=retest2 |

### Cycle 55 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 3485.60 | 3360.88 | 3345.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 14:15:00 | 3531.20 | 3442.57 | 3401.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 3475.00 | 3475.68 | 3445.76 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 3483.20 | 3477.58 | 3449.35 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:45:00 | 3483.10 | 3479.16 | 3455.08 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 14:45:00 | 3483.90 | 3475.18 | 3458.95 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 3455.20 | 3471.18 | 3458.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 3455.20 | 3471.18 | 3458.61 | SL hit (close<ema400) qty=1.00 sl=3458.61 alert=retest1 |

### Cycle 56 — SELL (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 14:15:00 | 3652.80 | 3662.30 | 3662.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 15:15:00 | 3630.90 | 3656.02 | 3659.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 3516.70 | 3513.72 | 3544.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 14:00:00 | 3516.70 | 3513.72 | 3544.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 3485.00 | 3466.67 | 3499.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 3514.80 | 3466.67 | 3499.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 3486.80 | 3470.70 | 3498.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 3428.00 | 3470.70 | 3498.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 3464.10 | 3455.01 | 3477.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 12:45:00 | 3465.70 | 3449.86 | 3465.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 3459.80 | 3456.67 | 3466.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3487.60 | 3463.23 | 3467.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 3467.20 | 3466.06 | 3468.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:15:00 | 3448.60 | 3464.16 | 3465.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 3292.41 | 3349.63 | 3381.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 09:15:00 | 3293.84 | 3349.63 | 3381.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 3290.89 | 3334.35 | 3371.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 3286.81 | 3334.35 | 3371.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 10:15:00 | 3276.17 | 3334.35 | 3371.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 12:15:00 | 3256.60 | 3307.24 | 3352.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 3287.00 | 3283.56 | 3328.04 | SL hit (close>ema200) qty=0.50 sl=3283.56 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 3047.80 | 3037.43 | 3036.98 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 3018.80 | 3033.70 | 3035.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 3012.20 | 3029.40 | 3033.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 3038.20 | 2974.17 | 2992.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 3038.20 | 2974.17 | 2992.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 3038.20 | 2974.17 | 2992.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 3038.20 | 2974.17 | 2992.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 3020.00 | 2983.34 | 2994.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 3290.00 | 2983.34 | 2994.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 3286.50 | 3043.97 | 3021.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 3428.90 | 3371.42 | 3325.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 3465.20 | 3488.44 | 3446.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 15:00:00 | 3465.20 | 3488.44 | 3446.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 3479.00 | 3481.99 | 3450.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 3455.50 | 3481.99 | 3450.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 3379.40 | 3465.40 | 3459.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:00:00 | 3379.40 | 3465.40 | 3459.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 3362.30 | 3444.78 | 3450.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 3348.90 | 3413.64 | 3434.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 14:15:00 | 3396.50 | 3357.10 | 3385.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 14:15:00 | 3396.50 | 3357.10 | 3385.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 14:15:00 | 3396.50 | 3357.10 | 3385.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 14:45:00 | 3400.00 | 3357.10 | 3385.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 15:15:00 | 3399.00 | 3365.48 | 3386.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 3340.30 | 3365.48 | 3386.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 3403.50 | 3364.45 | 3363.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 3403.50 | 3364.45 | 3363.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 14:15:00 | 3421.00 | 3398.31 | 3388.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 3371.50 | 3392.95 | 3387.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 3371.50 | 3392.95 | 3387.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3371.50 | 3392.95 | 3387.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:45:00 | 3430.00 | 3402.21 | 3392.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 14:30:00 | 3463.10 | 3427.66 | 3414.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 11:45:00 | 3425.10 | 3457.66 | 3456.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 3425.80 | 3457.66 | 3456.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 12:15:00 | 3430.00 | 3452.13 | 3454.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 3430.00 | 3452.13 | 3454.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 3404.60 | 3435.21 | 3444.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 13:15:00 | 3429.80 | 3426.26 | 3437.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 3429.80 | 3426.26 | 3437.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 3429.80 | 3426.26 | 3437.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:00:00 | 3429.80 | 3426.26 | 3437.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 3481.80 | 3437.37 | 3441.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 3505.00 | 3437.37 | 3441.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 15:15:00 | 3472.00 | 3444.29 | 3444.26 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 3397.00 | 3434.84 | 3439.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 3240.10 | 3340.99 | 3385.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 3338.00 | 3295.11 | 3340.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 15:00:00 | 3338.00 | 3295.11 | 3340.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 3348.00 | 3305.69 | 3340.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 09:15:00 | 3314.10 | 3305.69 | 3340.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 3148.39 | 3241.03 | 3275.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-09 14:15:00 | 3295.20 | 3214.29 | 3244.36 | SL hit (close>ema200) qty=0.50 sl=3214.29 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 3290.00 | 3258.73 | 3258.66 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 3234.90 | 3258.70 | 3261.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 3233.40 | 3253.64 | 3259.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-11 14:15:00 | 3259.00 | 3254.71 | 3259.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 3259.00 | 3254.71 | 3259.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 3259.00 | 3254.71 | 3259.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:45:00 | 3253.10 | 3254.71 | 3259.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3087.60 | 3046.51 | 3096.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 3107.90 | 3046.51 | 3096.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 3045.20 | 3046.25 | 3091.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 3008.00 | 3046.25 | 3091.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 3037.00 | 3042.83 | 3081.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:45:00 | 3044.60 | 3044.18 | 3079.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:30:00 | 3026.80 | 3039.44 | 3070.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3077.70 | 3032.82 | 3058.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 3088.00 | 3032.82 | 3058.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 3068.80 | 3040.01 | 3059.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 14:15:00 | 3123.10 | 3069.10 | 3068.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 3123.10 | 3069.10 | 3068.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 15:15:00 | 3177.80 | 3090.84 | 3078.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 14:15:00 | 3174.10 | 3184.70 | 3140.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 15:00:00 | 3174.10 | 3184.70 | 3140.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 3183.00 | 3184.36 | 3144.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 13:15:00 | 3185.00 | 3174.61 | 3151.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 14:45:00 | 3189.30 | 3172.69 | 3154.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 3109.30 | 3159.76 | 3151.89 | SL hit (close<static) qty=1.00 sl=3130.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 3087.70 | 3145.35 | 3146.05 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-23 14:15:00 | 3259.90 | 3153.03 | 3147.29 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 3088.00 | 3175.79 | 3184.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 3068.00 | 3140.91 | 3166.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 3040.90 | 3023.73 | 3081.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 15:00:00 | 3040.90 | 3023.73 | 3081.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3153.00 | 3046.10 | 3081.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 3153.00 | 3046.10 | 3081.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 3120.60 | 3061.00 | 3085.25 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 3181.00 | 3099.69 | 3099.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 13:15:00 | 3233.00 | 3126.35 | 3111.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3121.10 | 3149.08 | 3128.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3121.10 | 3149.08 | 3128.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3121.10 | 3149.08 | 3128.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 3116.60 | 3149.08 | 3128.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3127.40 | 3144.74 | 3128.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 3138.00 | 3144.74 | 3128.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 3318.40 | 3361.31 | 3363.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 09:15:00 | 3318.40 | 3361.31 | 3363.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 14:15:00 | 3281.00 | 3318.24 | 3339.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 11:15:00 | 3295.00 | 3281.89 | 3299.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 11:15:00 | 3295.00 | 3281.89 | 3299.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 3295.00 | 3281.89 | 3299.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 3295.00 | 3281.89 | 3299.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 3296.90 | 3284.89 | 3299.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 14:00:00 | 3278.60 | 3283.63 | 3297.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 3310.40 | 3284.39 | 3293.77 | SL hit (close>static) qty=1.00 sl=3300.00 alert=retest2 |

### Cycle 73 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 3324.90 | 3301.03 | 3299.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 14:15:00 | 3387.00 | 3328.36 | 3316.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 15:15:00 | 3395.00 | 3400.94 | 3370.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 09:30:00 | 3396.50 | 3400.57 | 3372.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 3389.80 | 3401.63 | 3383.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 3389.80 | 3401.63 | 3383.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 3392.00 | 3399.70 | 3384.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 3384.90 | 3399.70 | 3384.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 3385.90 | 3396.94 | 3384.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 3389.70 | 3396.94 | 3384.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 3358.70 | 3389.29 | 3382.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 3358.70 | 3389.29 | 3382.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 3368.10 | 3385.05 | 3381.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:45:00 | 3362.00 | 3385.05 | 3381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 3359.40 | 3377.39 | 3378.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 3350.00 | 3369.56 | 3374.28 | Break + close below crossover candle low |

### Cycle 75 — BUY (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 09:15:00 | 3415.80 | 3378.81 | 3378.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 3525.50 | 3416.14 | 3395.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 11:15:00 | 3733.20 | 3736.91 | 3686.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 12:00:00 | 3733.20 | 3736.91 | 3686.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 3772.10 | 3791.30 | 3767.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 3760.20 | 3791.30 | 3767.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 3775.20 | 3788.08 | 3768.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 3754.40 | 3788.08 | 3768.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 3769.10 | 3784.28 | 3768.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 3769.10 | 3784.28 | 3768.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 3764.70 | 3780.37 | 3768.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 3762.10 | 3780.37 | 3768.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 3771.10 | 3778.51 | 3768.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:15:00 | 3777.60 | 3778.51 | 3768.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 3777.60 | 3778.33 | 3769.14 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-21 10:15:00 | 3983.80 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-05-21 14:30:00 | 4001.60 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-05-22 09:30:00 | 3990.30 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-22 11:00:00 | 3993.90 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-05-23 14:45:00 | 4033.00 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-05-27 11:45:00 | 4037.00 | 2025-05-27 15:15:00 | 3949.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-06-26 09:15:00 | 3474.90 | 2025-07-02 10:15:00 | 3525.00 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2025-07-07 09:15:00 | 3494.40 | 2025-07-15 11:15:00 | 3437.20 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2025-07-07 09:45:00 | 3505.00 | 2025-07-15 11:15:00 | 3437.20 | STOP_HIT | 1.00 | 1.93% |
| SELL | retest2 | 2025-07-07 11:45:00 | 3510.00 | 2025-07-15 11:15:00 | 3437.20 | STOP_HIT | 1.00 | 2.07% |
| SELL | retest2 | 2025-07-07 15:15:00 | 3503.00 | 2025-07-15 11:15:00 | 3437.20 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2025-07-08 10:30:00 | 3461.30 | 2025-07-15 11:15:00 | 3437.20 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2025-07-24 13:45:00 | 3575.00 | 2025-07-25 09:15:00 | 3526.30 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-07-24 14:45:00 | 3585.20 | 2025-07-25 09:15:00 | 3526.30 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-31 12:15:00 | 3588.80 | 2025-08-01 11:15:00 | 3531.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-08-05 14:00:00 | 3443.60 | 2025-08-06 09:15:00 | 3646.90 | STOP_HIT | 1.00 | -5.90% |
| SELL | retest2 | 2025-08-13 10:45:00 | 3423.80 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-08-13 12:45:00 | 3426.00 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-13 13:15:00 | 3421.40 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-08-18 12:45:00 | 3420.00 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-08-19 14:00:00 | 3388.80 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-19 15:00:00 | 3383.50 | 2025-08-20 14:15:00 | 3419.10 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-08-29 11:15:00 | 3441.70 | 2025-08-29 15:15:00 | 3398.60 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-08-29 14:45:00 | 3428.90 | 2025-08-29 15:15:00 | 3398.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-09-04 12:00:00 | 3392.00 | 2025-09-08 09:15:00 | 3440.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-05 10:15:00 | 3391.90 | 2025-09-08 09:15:00 | 3440.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-09-05 10:45:00 | 3389.60 | 2025-09-08 09:15:00 | 3440.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-09-12 09:15:00 | 3671.00 | 2025-09-17 09:15:00 | 3616.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-09-23 15:15:00 | 3865.00 | 2025-09-24 12:15:00 | 3812.90 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-09-26 09:15:00 | 3736.60 | 2025-10-03 09:15:00 | 3738.60 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-10-20 09:15:00 | 3734.20 | 2025-10-24 12:15:00 | 3689.90 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-20 11:30:00 | 3760.00 | 2025-10-24 12:15:00 | 3689.90 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-10-20 12:30:00 | 3732.00 | 2025-10-24 12:15:00 | 3689.90 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-10-20 13:00:00 | 3732.80 | 2025-10-24 12:15:00 | 3689.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-10-27 09:15:00 | 3683.60 | 2025-10-31 09:15:00 | 3718.00 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-19 09:15:00 | 3553.10 | 2025-11-24 13:15:00 | 3375.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-19 09:15:00 | 3553.10 | 2025-11-24 15:15:00 | 3548.00 | STOP_HIT | 0.50 | 0.14% |
| SELL | retest2 | 2025-12-02 11:15:00 | 3407.30 | 2025-12-08 13:15:00 | 3236.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 09:15:00 | 3398.10 | 2025-12-08 13:15:00 | 3228.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 3395.50 | 2025-12-08 13:15:00 | 3225.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 11:15:00 | 3407.30 | 2025-12-08 14:15:00 | 3320.00 | STOP_HIT | 0.50 | 2.56% |
| SELL | retest2 | 2025-12-03 09:15:00 | 3398.10 | 2025-12-08 14:15:00 | 3320.00 | STOP_HIT | 0.50 | 2.30% |
| SELL | retest2 | 2025-12-05 14:30:00 | 3395.50 | 2025-12-08 14:15:00 | 3320.00 | STOP_HIT | 0.50 | 2.22% |
| SELL | retest2 | 2025-12-09 14:00:00 | 3377.80 | 2025-12-09 14:15:00 | 3485.60 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest1 | 2025-12-12 09:30:00 | 3483.20 | 2025-12-12 15:15:00 | 3455.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-12-12 11:45:00 | 3483.10 | 2025-12-12 15:15:00 | 3455.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-12-12 14:45:00 | 3483.90 | 2025-12-12 15:15:00 | 3455.20 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-12-15 14:45:00 | 3531.50 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2025-12-16 14:45:00 | 3524.90 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 3.63% |
| BUY | retest2 | 2025-12-17 14:30:00 | 3489.00 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 4.69% |
| BUY | retest2 | 2025-12-18 11:15:00 | 3494.80 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 4.52% |
| BUY | retest2 | 2025-12-23 14:30:00 | 3618.70 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2025-12-24 12:00:00 | 3607.40 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 1.26% |
| BUY | retest2 | 2025-12-24 12:30:00 | 3608.30 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 1.23% |
| BUY | retest2 | 2025-12-24 13:45:00 | 3614.80 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 1.05% |
| BUY | retest2 | 2025-12-26 14:15:00 | 3624.70 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-12-29 09:30:00 | 3622.10 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 0.85% |
| BUY | retest2 | 2025-12-30 13:30:00 | 3617.90 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-12-30 14:30:00 | 3625.30 | 2026-01-02 14:15:00 | 3652.80 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2026-01-09 09:15:00 | 3428.00 | 2026-01-20 09:15:00 | 3292.41 | PARTIAL | 0.50 | 3.96% |
| SELL | retest2 | 2026-01-09 14:45:00 | 3464.10 | 2026-01-20 09:15:00 | 3293.84 | PARTIAL | 0.50 | 4.91% |
| SELL | retest2 | 2026-01-12 12:45:00 | 3465.70 | 2026-01-20 10:15:00 | 3290.89 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-01-12 15:00:00 | 3459.80 | 2026-01-20 10:15:00 | 3286.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 3467.20 | 2026-01-20 10:15:00 | 3276.17 | PARTIAL | 0.50 | 5.51% |
| SELL | retest2 | 2026-01-14 12:15:00 | 3448.60 | 2026-01-20 12:15:00 | 3256.60 | PARTIAL | 0.50 | 5.57% |
| SELL | retest2 | 2026-01-09 09:15:00 | 3428.00 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 4.11% |
| SELL | retest2 | 2026-01-09 14:45:00 | 3464.10 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 5.11% |
| SELL | retest2 | 2026-01-12 12:45:00 | 3465.70 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 5.16% |
| SELL | retest2 | 2026-01-12 15:00:00 | 3459.80 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2026-01-13 11:45:00 | 3467.20 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 5.20% |
| SELL | retest2 | 2026-01-14 12:15:00 | 3448.60 | 2026-01-20 15:15:00 | 3287.00 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2026-02-16 09:15:00 | 3340.30 | 2026-02-17 13:15:00 | 3403.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-20 10:45:00 | 3430.00 | 2026-02-26 12:15:00 | 3430.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2026-02-23 14:30:00 | 3463.10 | 2026-02-26 12:15:00 | 3430.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2026-02-26 11:45:00 | 3425.10 | 2026-02-26 12:15:00 | 3430.00 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2026-02-26 12:15:00 | 3425.80 | 2026-02-26 12:15:00 | 3430.00 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-03-05 09:15:00 | 3314.10 | 2026-03-09 09:15:00 | 3148.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 09:15:00 | 3314.10 | 2026-03-09 14:15:00 | 3295.20 | STOP_HIT | 0.50 | 0.57% |
| SELL | retest2 | 2026-03-17 09:15:00 | 3008.00 | 2026-03-18 14:15:00 | 3123.10 | STOP_HIT | 1.00 | -3.83% |
| SELL | retest2 | 2026-03-17 11:15:00 | 3037.00 | 2026-03-18 14:15:00 | 3123.10 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-03-17 11:45:00 | 3044.60 | 2026-03-18 14:15:00 | 3123.10 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-17 13:30:00 | 3026.80 | 2026-03-18 14:15:00 | 3123.10 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-03-20 13:15:00 | 3185.00 | 2026-03-23 09:15:00 | 3109.30 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2026-03-20 14:45:00 | 3189.30 | 2026-03-23 09:15:00 | 3109.30 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2026-04-02 11:15:00 | 3138.00 | 2026-04-20 09:15:00 | 3318.40 | STOP_HIT | 1.00 | 5.75% |
| SELL | retest2 | 2026-04-22 14:00:00 | 3278.60 | 2026-04-23 09:15:00 | 3310.40 | STOP_HIT | 1.00 | -0.97% |
