# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 3955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 8 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 37 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 33 |
| PARTIAL | 1 |
| TARGET_HIT | 8 |
| STOP_HIT | 29 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 34 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 29
- **Target hits / Stop hits / Partials:** 4 / 29 / 1
- **Avg / median % per leg:** -0.27% / -1.47%
- **Sum % (uncompounded):** -9.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 3 | 12.0% | 3 | 22 | 0 | -0.33% | -8.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 3 | 12.0% | 3 | 22 | 0 | -0.33% | -8.3% |
| SELL (all) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.11% | -0.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.11% | -0.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 34 | 5 | 14.7% | 4 | 29 | 1 | -0.27% | -9.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 09:15:00 | 3620.00 | 3765.59 | 3765.75 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 14:15:00 | 3993.60 | 3758.77 | 3757.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 09:15:00 | 3999.95 | 3763.20 | 3760.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 10:15:00 | 3859.85 | 3873.24 | 3823.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-15 10:30:00 | 3861.30 | 3873.24 | 3823.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 14:15:00 | 3802.10 | 3871.72 | 3824.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 15:00:00 | 3802.10 | 3871.72 | 3824.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 3819.00 | 3871.20 | 3824.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:15:00 | 3837.00 | 3871.20 | 3824.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 3841.25 | 3870.90 | 3824.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 3885.30 | 3864.11 | 3825.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 12:00:00 | 3864.25 | 3863.64 | 3825.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 09:15:00 | 3871.50 | 3861.37 | 3826.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 12:45:00 | 3868.50 | 3864.09 | 3829.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 3796.50 | 3867.31 | 3833.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 3796.50 | 3867.31 | 3833.04 | SL hit (close<static) qty=1.00 sl=3800.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 13:15:00 | 3766.25 | 3812.60 | 3812.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 11:15:00 | 3747.30 | 3809.91 | 3811.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 15:15:00 | 3739.00 | 3733.45 | 3763.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-10 09:15:00 | 3750.05 | 3733.45 | 3763.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 3756.65 | 3733.68 | 3763.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 10:00:00 | 3756.65 | 3733.68 | 3763.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 3788.00 | 3734.22 | 3763.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 3788.00 | 3734.22 | 3763.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 3780.00 | 3734.67 | 3763.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-10 12:15:00 | 3768.70 | 3734.67 | 3763.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 3758.00 | 3736.27 | 3764.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 10:15:00 | 3803.60 | 3737.08 | 3764.30 | SL hit (close>static) qty=1.00 sl=3791.95 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 4119.00 | 3786.45 | 3786.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 4175.00 | 3790.32 | 3788.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 13:15:00 | 4449.55 | 4458.30 | 4298.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-21 14:00:00 | 4449.55 | 4458.30 | 4298.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 4309.50 | 4430.51 | 4317.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 13:45:00 | 4303.55 | 4430.51 | 4317.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 4298.90 | 4429.20 | 4317.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 4298.90 | 4429.20 | 4317.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 4302.00 | 4427.93 | 4317.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 4328.15 | 4427.93 | 4317.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 4316.60 | 4425.93 | 4317.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 4316.60 | 4425.93 | 4317.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 4299.95 | 4424.68 | 4317.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 4299.95 | 4424.68 | 4317.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 4299.20 | 4423.43 | 4317.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 4295.60 | 4423.43 | 4317.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 4320.00 | 4421.38 | 4317.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:30:00 | 4309.40 | 4421.38 | 4317.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 4320.00 | 4420.37 | 4317.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 4358.65 | 4420.37 | 4317.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 15:00:00 | 4350.60 | 4411.07 | 4325.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 4256.55 | 4408.93 | 4325.57 | SL hit (close<static) qty=1.00 sl=4317.10 alert=retest2 |

### Cycle 5 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 4100.00 | 4301.73 | 4302.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 4094.25 | 4290.46 | 4296.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 4249.10 | 4244.70 | 4270.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 12:15:00 | 4249.10 | 4244.70 | 4270.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 12:15:00 | 4249.10 | 4244.70 | 4270.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 12:30:00 | 4244.75 | 4244.70 | 4270.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 4230.00 | 4244.51 | 4269.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 4169.90 | 4244.51 | 4269.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 3961.40 | 4220.24 | 4254.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-30 13:15:00 | 3752.91 | 4113.12 | 4189.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 3502.20 | 3286.12 | 3285.53 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 3176.60 | 3345.00 | 3345.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3163.00 | 3328.80 | 3337.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 3108.40 | 3107.12 | 3171.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:30:00 | 3102.90 | 3107.12 | 3171.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3148.70 | 3095.54 | 3143.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 3148.70 | 3095.54 | 3143.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3129.00 | 3095.87 | 3143.12 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 3310.00 | 3172.80 | 3172.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 15:15:00 | 3315.00 | 3178.15 | 3175.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 10:15:00 | 3243.30 | 3243.56 | 3214.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:45:00 | 3240.20 | 3243.56 | 3214.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3228.50 | 3253.33 | 3222.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3239.60 | 3253.33 | 3222.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3255.40 | 3253.35 | 3222.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 3263.70 | 3253.43 | 3222.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 15:15:00 | 3260.00 | 3253.39 | 3222.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-12 09:15:00 | 3590.07 | 3287.54 | 3242.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 3703.80 | 3818.26 | 3818.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 3678.00 | 3814.77 | 3816.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 3890.50 | 3795.94 | 3806.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3825.20 | 3796.23 | 3807.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3863.00 | 3796.23 | 3807.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3677.00 | 3665.92 | 3721.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 3673.70 | 3665.92 | 3721.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 3670.00 | 3666.58 | 3720.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 3655.00 | 3666.62 | 3720.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 3655.10 | 3667.18 | 3719.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 3780.90 | 3667.60 | 3718.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 3780.90 | 3667.60 | 3718.97 | SL hit (close>static) qty=1.00 sl=3778.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 4015.80 | 3759.10 | 3758.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 4044.80 | 3764.49 | 3761.29 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-30 09:30:00 | 2834.40 | 2023-05-30 14:15:00 | 3006.19 | TARGET_HIT | 1.00 | 6.06% |
| BUY | retest2 | 2024-04-22 09:15:00 | 3885.30 | 2024-04-29 09:15:00 | 3796.50 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2024-04-22 12:00:00 | 3864.25 | 2024-04-29 09:15:00 | 3796.50 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-04-24 09:15:00 | 3871.50 | 2024-04-29 09:15:00 | 3796.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2024-04-25 12:45:00 | 3868.50 | 2024-04-29 09:15:00 | 3796.50 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-04-30 09:15:00 | 3834.45 | 2024-05-02 14:15:00 | 3775.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-05-02 11:00:00 | 3805.15 | 2024-05-02 14:15:00 | 3775.05 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-05-07 09:15:00 | 3834.10 | 2024-05-08 10:15:00 | 3777.55 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-05-07 13:30:00 | 3805.15 | 2024-05-08 10:15:00 | 3777.55 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-05-14 14:30:00 | 3820.65 | 2024-05-15 09:15:00 | 3720.00 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-05-14 15:15:00 | 3820.00 | 2024-05-15 09:15:00 | 3720.00 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-05-16 10:00:00 | 3827.80 | 2024-05-16 13:15:00 | 3743.85 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2024-06-10 12:15:00 | 3768.70 | 2024-06-11 10:15:00 | 3803.60 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-06-11 09:15:00 | 3758.00 | 2024-06-11 10:15:00 | 3803.60 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-06-12 09:45:00 | 3771.00 | 2024-06-12 11:15:00 | 3808.05 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-03 09:15:00 | 4358.65 | 2024-09-09 09:15:00 | 4256.55 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-09-06 15:00:00 | 4350.60 | 2024-09-09 09:15:00 | 4256.55 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-09-10 11:45:00 | 4327.30 | 2024-09-11 14:15:00 | 4315.60 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-09-10 12:30:00 | 4325.20 | 2024-09-11 14:15:00 | 4315.60 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-09-10 14:15:00 | 4355.25 | 2024-09-11 14:15:00 | 4315.60 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-09-11 12:45:00 | 4351.05 | 2024-09-11 14:15:00 | 4315.60 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-09-13 10:00:00 | 4350.00 | 2024-09-16 09:15:00 | 4314.40 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-09-24 09:30:00 | 4356.05 | 2024-09-25 11:15:00 | 4301.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-09-24 11:15:00 | 4390.00 | 2024-09-25 11:15:00 | 4301.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-09-27 12:00:00 | 4376.85 | 2024-09-27 14:15:00 | 4283.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-10-01 10:00:00 | 4378.05 | 2024-10-01 13:15:00 | 4310.80 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-10-18 09:15:00 | 4169.90 | 2024-10-22 14:15:00 | 3961.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 09:15:00 | 4169.90 | 2024-10-30 13:15:00 | 3752.91 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-11-07 11:30:00 | 3263.70 | 2025-11-12 09:15:00 | 3590.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 15:15:00 | 3260.00 | 2025-11-12 09:15:00 | 3586.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 10:15:00 | 3673.70 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-08 15:15:00 | 3670.00 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-04-09 09:30:00 | 3655.00 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-04-09 13:15:00 | 3655.10 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.44% |
