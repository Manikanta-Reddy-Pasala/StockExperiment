# Pfizer Ltd. (PFIZER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4793.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 144 |
| ALERT2 | 140 |
| ALERT2_SKIP | 101 |
| ALERT3 | 284 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 122 |
| PARTIAL | 9 |
| TARGET_HIT | 1 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 96
- **Target hits / Stop hits / Partials:** 1 / 127 / 9
- **Avg / median % per leg:** 0.04% / -0.88%
- **Sum % (uncompounded):** 5.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 73 | 16 | 21.9% | 0 | 73 | 0 | -0.51% | -37.1% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.97% | -4.8% |
| BUY @ 3rd Alert (retest2) | 68 | 16 | 23.5% | 0 | 68 | 0 | -0.47% | -32.3% |
| SELL (all) | 64 | 25 | 39.1% | 1 | 54 | 9 | 0.67% | 43.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.06% | -1.1% |
| SELL @ 3rd Alert (retest2) | 63 | 25 | 39.7% | 1 | 53 | 9 | 0.70% | 44.1% |
| retest1 (combined) | 6 | 0 | 0.0% | 0 | 6 | 0 | -0.98% | -5.9% |
| retest2 (combined) | 131 | 41 | 31.3% | 1 | 121 | 9 | 0.09% | 11.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 3805.00 | 3816.15 | 3816.95 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 12:15:00 | 3824.45 | 3818.09 | 3817.64 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-22 13:15:00 | 3817.00 | 3818.35 | 3818.48 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 14:15:00 | 3835.90 | 3821.86 | 3820.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 15:15:00 | 3851.20 | 3827.73 | 3822.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 13:15:00 | 3832.00 | 3837.86 | 3831.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 13:15:00 | 3832.00 | 3837.86 | 3831.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 13:15:00 | 3832.00 | 3837.86 | 3831.16 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 3820.00 | 3828.73 | 3829.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 3817.00 | 3826.28 | 3828.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 14:15:00 | 3826.00 | 3817.53 | 3822.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 14:15:00 | 3826.00 | 3817.53 | 3822.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 14:15:00 | 3826.00 | 3817.53 | 3822.20 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 3833.30 | 3806.22 | 3805.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 13:15:00 | 3890.05 | 3838.28 | 3824.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-06 11:15:00 | 3938.95 | 3947.28 | 3910.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 14:15:00 | 3974.75 | 3962.84 | 3942.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 14:15:00 | 3974.75 | 3962.84 | 3942.33 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 10:15:00 | 3913.20 | 3946.84 | 3951.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-12 12:15:00 | 3891.25 | 3930.98 | 3943.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-14 09:15:00 | 3887.70 | 3880.52 | 3901.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-14 09:15:00 | 3887.70 | 3880.52 | 3901.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 3887.70 | 3880.52 | 3901.78 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 13:15:00 | 3803.10 | 3786.50 | 3784.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 14:15:00 | 3833.30 | 3808.33 | 3798.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 11:15:00 | 3805.00 | 3818.11 | 3807.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 11:15:00 | 3805.00 | 3818.11 | 3807.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 11:15:00 | 3805.00 | 3818.11 | 3807.30 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 15:15:00 | 3782.00 | 3798.22 | 3800.21 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 10:15:00 | 3826.00 | 3800.70 | 3798.47 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 15:15:00 | 3785.00 | 3799.24 | 3799.53 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 10:15:00 | 3806.95 | 3800.57 | 3800.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 11:15:00 | 3811.75 | 3802.80 | 3801.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-05 14:15:00 | 3800.00 | 3802.75 | 3801.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-05 14:15:00 | 3800.00 | 3802.75 | 3801.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 14:15:00 | 3800.00 | 3802.75 | 3801.56 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 15:15:00 | 3787.00 | 3799.60 | 3800.24 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 3814.20 | 3802.52 | 3801.51 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-06 14:15:00 | 3786.00 | 3801.10 | 3801.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-06 15:15:00 | 3773.00 | 3795.48 | 3799.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-07 09:15:00 | 3797.95 | 3795.97 | 3799.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 3797.95 | 3795.97 | 3799.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 3797.95 | 3795.97 | 3799.01 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 14:15:00 | 3796.95 | 3794.11 | 3794.07 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 15:15:00 | 3785.00 | 3792.29 | 3793.25 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 3818.50 | 3797.53 | 3795.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-12 10:15:00 | 3825.05 | 3803.04 | 3798.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 15:15:00 | 3805.00 | 3810.66 | 3804.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 15:15:00 | 3805.00 | 3810.66 | 3804.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 15:15:00 | 3805.00 | 3810.66 | 3804.74 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 15:15:00 | 3781.00 | 3801.90 | 3804.18 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 3821.40 | 3802.81 | 3802.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-17 11:15:00 | 3840.70 | 3814.80 | 3808.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 14:15:00 | 3864.90 | 3867.66 | 3848.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-19 10:15:00 | 3866.40 | 3867.87 | 3853.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 3866.40 | 3867.87 | 3853.14 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 13:15:00 | 3852.30 | 3872.68 | 3875.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 3847.15 | 3867.57 | 3872.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 09:15:00 | 3865.15 | 3862.93 | 3869.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 09:15:00 | 3865.15 | 3862.93 | 3869.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 09:15:00 | 3865.15 | 3862.93 | 3869.48 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 12:15:00 | 3920.00 | 3870.51 | 3864.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 10:15:00 | 3946.55 | 3902.66 | 3883.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 09:15:00 | 4002.70 | 4014.12 | 3991.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 4002.70 | 4014.12 | 3991.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 4002.70 | 4014.12 | 3991.15 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 3978.80 | 3993.27 | 3993.52 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 11:15:00 | 3995.00 | 3992.91 | 3992.86 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 12:15:00 | 3968.15 | 3987.96 | 3990.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-07 13:15:00 | 3943.35 | 3965.74 | 3975.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-08 09:15:00 | 3988.20 | 3964.01 | 3971.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-08 09:15:00 | 3988.20 | 3964.01 | 3971.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 3988.20 | 3964.01 | 3971.99 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 12:15:00 | 3988.00 | 3971.26 | 3969.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-09 14:15:00 | 4030.00 | 3985.04 | 3975.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-10 13:15:00 | 4012.75 | 4035.66 | 4011.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 13:15:00 | 4012.75 | 4035.66 | 4011.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 13:15:00 | 4012.75 | 4035.66 | 4011.03 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 13:15:00 | 3980.00 | 4007.71 | 4008.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 3967.80 | 3999.73 | 4004.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 3888.00 | 3877.46 | 3921.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 09:15:00 | 3869.00 | 3866.62 | 3892.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 09:15:00 | 3869.00 | 3866.62 | 3892.55 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 14:15:00 | 3913.45 | 3885.29 | 3884.42 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 3872.50 | 3893.28 | 3894.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-24 09:15:00 | 3869.00 | 3888.43 | 3892.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 14:15:00 | 3845.45 | 3845.15 | 3854.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 3853.55 | 3847.60 | 3853.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 3853.55 | 3847.60 | 3853.99 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 3856.05 | 3832.76 | 3831.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 09:15:00 | 3889.95 | 3857.56 | 3845.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 14:15:00 | 3869.35 | 3869.89 | 3857.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 13:15:00 | 3865.40 | 3875.10 | 3866.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 3865.40 | 3875.10 | 3866.42 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 11:15:00 | 3852.45 | 3860.39 | 3861.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 15:15:00 | 3842.85 | 3855.15 | 3858.51 | Break + close below crossover candle low |

### Cycle 32 — BUY (started 2023-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-08 09:15:00 | 3891.15 | 3862.35 | 3861.48 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 15:15:00 | 3847.00 | 3861.69 | 3862.54 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 09:15:00 | 3870.10 | 3863.37 | 3863.23 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2023-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-11 11:15:00 | 3848.35 | 3861.38 | 3862.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 15:15:00 | 3840.50 | 3849.16 | 3854.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 10:15:00 | 3860.00 | 3850.94 | 3854.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 10:15:00 | 3860.00 | 3850.94 | 3854.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 3860.00 | 3850.94 | 3854.18 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 11:15:00 | 3875.80 | 3856.69 | 3854.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 14:15:00 | 3914.90 | 3870.77 | 3861.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-20 09:15:00 | 3874.00 | 3894.24 | 3884.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 3874.00 | 3894.24 | 3884.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 3874.00 | 3894.24 | 3884.08 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 3843.20 | 3874.76 | 3878.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 10:15:00 | 3838.95 | 3867.59 | 3874.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 12:15:00 | 3864.00 | 3844.51 | 3853.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-22 12:15:00 | 3864.00 | 3844.51 | 3853.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 3864.00 | 3844.51 | 3853.72 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 10:15:00 | 3868.00 | 3854.75 | 3854.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 12:15:00 | 3871.00 | 3859.66 | 3856.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 14:15:00 | 3860.90 | 3861.56 | 3858.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 14:15:00 | 3860.90 | 3861.56 | 3858.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 3860.90 | 3861.56 | 3858.34 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-09-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 09:15:00 | 3847.30 | 3855.82 | 3856.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 10:15:00 | 3840.45 | 3852.75 | 3855.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-28 11:15:00 | 3857.25 | 3853.65 | 3855.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 11:15:00 | 3857.25 | 3853.65 | 3855.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 3857.25 | 3853.65 | 3855.39 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 3875.15 | 3857.86 | 3856.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-03 14:15:00 | 3913.00 | 3893.34 | 3878.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 3953.40 | 3974.46 | 3956.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 3953.40 | 3974.46 | 3956.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 3953.40 | 3974.46 | 3956.60 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 12:15:00 | 3940.00 | 3955.05 | 3955.51 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 13:15:00 | 3961.00 | 3956.24 | 3956.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 14:15:00 | 3965.85 | 3958.16 | 3956.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-10 15:15:00 | 3948.00 | 3956.13 | 3956.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-10 15:15:00 | 3948.00 | 3956.13 | 3956.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 15:15:00 | 3948.00 | 3956.13 | 3956.10 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 11:15:00 | 3964.70 | 3969.56 | 3969.84 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 12:15:00 | 3985.00 | 3972.64 | 3971.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 13:15:00 | 3990.00 | 3976.12 | 3972.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 15:15:00 | 3970.00 | 3976.87 | 3973.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 15:15:00 | 3970.00 | 3976.87 | 3973.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 15:15:00 | 3970.00 | 3976.87 | 3973.95 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-16 14:15:00 | 3941.30 | 3972.90 | 3974.15 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 11:15:00 | 3994.00 | 3971.95 | 3970.70 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 3964.00 | 3971.06 | 3971.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 3935.85 | 3956.24 | 3964.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 15:15:00 | 3954.00 | 3951.86 | 3959.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 15:15:00 | 3954.00 | 3951.86 | 3959.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 3954.00 | 3951.86 | 3959.06 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 09:15:00 | 3970.80 | 3945.70 | 3944.81 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2023-10-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 12:15:00 | 3932.40 | 3943.02 | 3943.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-27 09:15:00 | 3923.00 | 3934.95 | 3939.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 10:15:00 | 3938.00 | 3935.56 | 3939.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 10:15:00 | 3938.00 | 3935.56 | 3939.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 3938.00 | 3935.56 | 3939.31 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 14:15:00 | 3954.75 | 3937.26 | 3936.15 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 09:15:00 | 3932.00 | 3935.41 | 3935.46 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 10:15:00 | 3948.95 | 3938.12 | 3936.68 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2023-11-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 09:15:00 | 3918.85 | 3933.96 | 3935.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 12:15:00 | 3880.05 | 3917.36 | 3926.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 09:15:00 | 3900.00 | 3897.03 | 3912.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 15:15:00 | 3910.00 | 3899.40 | 3906.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 3910.00 | 3899.40 | 3906.78 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 15:15:00 | 3930.00 | 3895.78 | 3891.48 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2023-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 13:15:00 | 3889.20 | 3896.93 | 3897.70 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 3904.05 | 3898.93 | 3898.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 3973.85 | 3926.88 | 3915.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 09:15:00 | 3951.90 | 3958.38 | 3941.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 10:15:00 | 4068.50 | 4109.44 | 4091.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 4068.50 | 4109.44 | 4091.14 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 09:15:00 | 4073.40 | 4082.39 | 4083.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 15:15:00 | 4034.00 | 4068.61 | 4076.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 10:15:00 | 4060.00 | 4046.51 | 4055.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-30 10:15:00 | 4060.00 | 4046.51 | 4055.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 4060.00 | 4046.51 | 4055.71 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 4083.85 | 4063.85 | 4061.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 4107.15 | 4075.89 | 4067.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 09:15:00 | 4060.55 | 4088.94 | 4080.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 09:15:00 | 4060.55 | 4088.94 | 4080.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 4060.55 | 4088.94 | 4080.93 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2023-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 12:15:00 | 4053.80 | 4072.44 | 4074.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-04 13:15:00 | 4045.85 | 4067.12 | 4071.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 14:15:00 | 4041.00 | 4027.37 | 4039.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 14:15:00 | 4041.00 | 4027.37 | 4039.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 14:15:00 | 4041.00 | 4027.37 | 4039.27 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 15:15:00 | 4052.60 | 4041.23 | 4040.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-08 09:15:00 | 4071.40 | 4047.27 | 4043.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 4052.00 | 4078.63 | 4071.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 4052.00 | 4078.63 | 4071.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 4052.00 | 4078.63 | 4071.35 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-22 12:15:00 | 4196.35 | 4220.01 | 4222.36 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2023-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 10:15:00 | 4233.30 | 4222.65 | 4222.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 12:15:00 | 4239.70 | 4227.24 | 4224.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 15:15:00 | 4250.00 | 4264.68 | 4254.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 15:15:00 | 4250.00 | 4264.68 | 4254.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 4250.00 | 4264.68 | 4254.59 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-09 13:15:00 | 4371.40 | 4376.79 | 4377.20 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 15:15:00 | 4387.00 | 4378.68 | 4377.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 10:15:00 | 4400.00 | 4383.69 | 4380.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 10:15:00 | 4388.40 | 4396.85 | 4390.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 10:15:00 | 4388.40 | 4396.85 | 4390.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 10:15:00 | 4388.40 | 4396.85 | 4390.64 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 13:15:00 | 4369.05 | 4385.61 | 4386.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-11 14:15:00 | 4335.40 | 4375.57 | 4381.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-12 12:15:00 | 4357.00 | 4355.87 | 4368.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 4359.30 | 4355.69 | 4364.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 4359.30 | 4355.69 | 4364.08 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-01-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 12:15:00 | 4260.00 | 4229.57 | 4227.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-25 13:15:00 | 4298.00 | 4243.26 | 4233.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 15:15:00 | 4344.70 | 4346.73 | 4321.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-30 15:15:00 | 4344.70 | 4346.73 | 4321.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 4344.70 | 4346.73 | 4321.42 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 14:15:00 | 4601.20 | 4672.73 | 4680.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 4563.00 | 4627.45 | 4655.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-12 09:15:00 | 4443.20 | 4441.69 | 4509.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 4443.20 | 4441.69 | 4509.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 4443.20 | 4441.69 | 4509.93 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 13:15:00 | 4480.55 | 4420.88 | 4417.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 4557.90 | 4468.79 | 4441.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-16 10:15:00 | 4529.15 | 4534.13 | 4498.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 11:15:00 | 4491.90 | 4525.68 | 4498.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 11:15:00 | 4491.90 | 4525.68 | 4498.19 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 4520.90 | 4583.76 | 4584.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 15:15:00 | 4497.00 | 4548.78 | 4566.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 4399.00 | 4386.76 | 4418.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 4472.25 | 4404.38 | 4420.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 4472.25 | 4404.38 | 4420.78 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 11:15:00 | 4485.00 | 4433.91 | 4432.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-28 14:15:00 | 4531.85 | 4467.51 | 4449.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 15:15:00 | 4568.00 | 4575.45 | 4528.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 4508.05 | 4566.74 | 4551.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 4508.05 | 4566.74 | 4551.84 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 10:15:00 | 4510.00 | 4543.83 | 4543.98 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-04 14:15:00 | 4569.65 | 4544.03 | 4543.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-04 15:15:00 | 4600.00 | 4555.22 | 4548.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-06 09:15:00 | 4555.45 | 4606.97 | 4589.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 09:15:00 | 4555.45 | 4606.97 | 4589.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 09:15:00 | 4555.45 | 4606.97 | 4589.28 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2024-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 13:15:00 | 4555.05 | 4575.93 | 4578.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 14:15:00 | 4533.95 | 4567.53 | 4574.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 4592.00 | 4569.46 | 4573.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 4592.00 | 4569.46 | 4573.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 4592.00 | 4569.46 | 4573.81 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 09:15:00 | 4330.75 | 4303.56 | 4301.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 15:15:00 | 4399.65 | 4360.80 | 4345.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-21 14:15:00 | 4358.50 | 4377.74 | 4363.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 14:15:00 | 4358.50 | 4377.74 | 4363.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 14:15:00 | 4358.50 | 4377.74 | 4363.09 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-03-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 10:15:00 | 4324.00 | 4355.47 | 4355.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 09:15:00 | 4259.00 | 4332.63 | 4343.72 | Break + close below crossover candle low |

### Cycle 76 — BUY (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-27 15:15:00 | 5000.00 | 4374.49 | 4329.26 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 4210.25 | 4306.98 | 4316.54 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 11:15:00 | 4367.90 | 4320.25 | 4318.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 13:15:00 | 4397.80 | 4343.40 | 4329.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 12:15:00 | 4385.05 | 4394.93 | 4367.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-02 14:15:00 | 4339.00 | 4383.25 | 4367.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 4339.00 | 4383.25 | 4367.21 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 15:15:00 | 4335.00 | 4364.63 | 4365.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 14:15:00 | 4322.45 | 4344.39 | 4353.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 15:15:00 | 4345.00 | 4344.51 | 4352.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 09:15:00 | 4382.05 | 4352.02 | 4355.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 09:15:00 | 4382.05 | 4352.02 | 4355.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 4202.05 | 4203.98 | 4240.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 4143.00 | 4121.48 | 4119.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 4143.00 | 4121.48 | 4119.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 09:15:00 | 4160.00 | 4135.70 | 4127.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 13:15:00 | 4141.10 | 4142.21 | 4133.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 13:30:00 | 4146.35 | 4142.21 | 4133.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 4143.40 | 4149.94 | 4141.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 11:45:00 | 4139.10 | 4149.94 | 4141.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 12:15:00 | 4123.25 | 4144.60 | 4139.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 13:00:00 | 4123.25 | 4144.60 | 4139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 4142.80 | 4144.24 | 4140.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-24 14:45:00 | 4158.65 | 4145.83 | 4141.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 11:30:00 | 4144.00 | 4146.24 | 4142.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 12:15:00 | 4105.15 | 4138.02 | 4139.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 12:15:00 | 4105.15 | 4138.02 | 4139.42 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 4148.40 | 4139.37 | 4139.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 11:15:00 | 4168.60 | 4147.74 | 4143.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 12:15:00 | 4138.40 | 4145.87 | 4142.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 12:15:00 | 4138.40 | 4145.87 | 4142.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 12:15:00 | 4138.40 | 4145.87 | 4142.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:00:00 | 4138.40 | 4145.87 | 4142.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 13:15:00 | 4141.90 | 4145.08 | 4142.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 13:45:00 | 4138.15 | 4145.08 | 4142.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 14:15:00 | 4189.00 | 4153.86 | 4146.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 15:15:00 | 4215.20 | 4153.86 | 4146.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 10:45:00 | 4199.95 | 4182.31 | 4163.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:15:00 | 4199.90 | 4179.28 | 4167.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 13:15:00 | 4200.40 | 4233.43 | 4235.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 13:15:00 | 4200.40 | 4233.43 | 4235.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 4191.00 | 4218.92 | 4227.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-06 11:15:00 | 4269.00 | 4227.51 | 4229.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-06 11:15:00 | 4269.00 | 4227.51 | 4229.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 11:15:00 | 4269.00 | 4227.51 | 4229.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 12:00:00 | 4269.00 | 4227.51 | 4229.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2024-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 12:15:00 | 4269.00 | 4235.81 | 4233.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 13:15:00 | 4276.20 | 4243.88 | 4237.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 15:15:00 | 4250.00 | 4267.68 | 4256.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 15:15:00 | 4250.00 | 4267.68 | 4256.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 4250.00 | 4267.68 | 4256.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 09:30:00 | 4249.55 | 4267.78 | 4257.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 4281.80 | 4270.59 | 4260.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 15:15:00 | 4287.75 | 4276.65 | 4266.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-09 09:30:00 | 4299.95 | 4284.78 | 4272.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 12:00:00 | 4285.90 | 4278.61 | 4276.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-10 12:30:00 | 4289.95 | 4279.89 | 4277.66 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 13:15:00 | 4336.70 | 4291.25 | 4283.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-10 13:30:00 | 4272.15 | 4291.25 | 4283.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 4300.00 | 4302.10 | 4289.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 12:30:00 | 4365.00 | 4328.44 | 4307.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-13 14:30:00 | 4378.95 | 4338.84 | 4316.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 09:30:00 | 4369.65 | 4347.46 | 4324.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 12:30:00 | 4370.90 | 4363.21 | 4337.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 4343.55 | 4360.75 | 4341.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 4343.55 | 4360.75 | 4341.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 4309.80 | 4350.56 | 4338.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 09:30:00 | 4309.80 | 4342.85 | 4336.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 4311.30 | 4336.54 | 4333.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:00:00 | 4311.30 | 4336.54 | 4333.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-05-15 11:15:00 | 4300.00 | 4329.23 | 4330.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 4300.00 | 4329.23 | 4330.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 15:15:00 | 4280.45 | 4313.56 | 4322.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 4334.15 | 4317.67 | 4323.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 09:15:00 | 4334.15 | 4317.67 | 4323.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 4334.15 | 4317.67 | 4323.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 10:00:00 | 4334.15 | 4317.67 | 4323.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 4322.15 | 4318.57 | 4323.14 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 14:15:00 | 4331.35 | 4325.36 | 4325.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 4398.70 | 4342.68 | 4333.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 11:15:00 | 4339.85 | 4349.69 | 4338.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 11:15:00 | 4339.85 | 4349.69 | 4338.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 4339.85 | 4349.69 | 4338.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 4339.85 | 4349.69 | 4338.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 4348.95 | 4349.54 | 4339.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:45:00 | 4327.85 | 4349.54 | 4339.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 4397.50 | 4359.13 | 4344.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:45:00 | 4349.90 | 4359.13 | 4344.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 4475.65 | 4429.17 | 4390.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 4500.05 | 4429.17 | 4390.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-21 12:15:00 | 4336.50 | 4405.30 | 4388.77 | SL hit (close<static) qty=1.00 sl=4370.80 alert=retest2 |

### Cycle 87 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 4530.00 | 4568.11 | 4569.10 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-05-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 12:15:00 | 4588.00 | 4569.99 | 4569.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 13:15:00 | 4609.15 | 4577.82 | 4573.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 4710.05 | 4727.14 | 4677.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 11:00:00 | 4710.05 | 4727.14 | 4677.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 4720.00 | 4718.03 | 4681.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 4589.90 | 4718.03 | 4681.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 4687.00 | 4723.18 | 4690.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 15:00:00 | 4687.00 | 4723.18 | 4690.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 4685.00 | 4715.54 | 4690.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-05 09:15:00 | 4777.00 | 4715.54 | 4690.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 4900.00 | 4915.52 | 4916.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 14:15:00 | 4900.00 | 4915.52 | 4916.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 10:15:00 | 4875.00 | 4902.21 | 4909.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-19 09:15:00 | 4810.05 | 4784.68 | 4815.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 09:15:00 | 4810.05 | 4784.68 | 4815.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 4810.05 | 4784.68 | 4815.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:00:00 | 4810.05 | 4784.68 | 4815.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 4859.40 | 4799.63 | 4819.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 4869.70 | 4799.63 | 4819.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 4876.20 | 4814.94 | 4824.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:45:00 | 4887.25 | 4814.94 | 4824.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 4835.00 | 4827.32 | 4828.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 4850.00 | 4827.32 | 4828.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 4850.75 | 4832.01 | 4830.29 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 4796.05 | 4826.74 | 4830.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 15:15:00 | 4781.00 | 4817.60 | 4825.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 4818.80 | 4817.84 | 4824.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 4818.80 | 4817.84 | 4824.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 4818.80 | 4817.84 | 4824.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:15:00 | 4790.00 | 4817.01 | 4823.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 14:15:00 | 4764.95 | 4811.95 | 4819.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 14:15:00 | 4550.50 | 4601.20 | 4624.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 14:15:00 | 4526.70 | 4601.20 | 4624.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 4612.45 | 4592.22 | 4615.49 | SL hit (close>ema200) qty=0.50 sl=4592.22 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 14:15:00 | 4609.10 | 4592.44 | 4591.37 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 15:15:00 | 4571.00 | 4588.15 | 4589.52 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 4612.60 | 4593.04 | 4591.62 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 15:15:00 | 4580.00 | 4595.53 | 4596.40 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 09:15:00 | 4612.55 | 4598.93 | 4597.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 4622.80 | 4603.71 | 4600.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 4618.55 | 4620.43 | 4610.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 14:45:00 | 4618.95 | 4620.43 | 4610.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 4622.00 | 4620.75 | 4611.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 4611.30 | 4620.75 | 4611.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 4606.60 | 4617.92 | 4611.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 10:15:00 | 4613.70 | 4617.92 | 4611.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 4617.05 | 4617.74 | 4611.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 4622.00 | 4617.74 | 4611.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 12:00:00 | 4629.00 | 4620.00 | 4613.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:45:00 | 4621.45 | 4622.75 | 4616.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 4588.05 | 4613.31 | 4613.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 4588.05 | 4613.31 | 4613.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 4584.90 | 4607.63 | 4610.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 4601.20 | 4598.18 | 4603.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 4601.20 | 4598.18 | 4603.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 4601.20 | 4598.18 | 4603.97 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 11:15:00 | 4667.45 | 4615.76 | 4611.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 12:15:00 | 4732.40 | 4639.08 | 4622.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 4941.50 | 4951.24 | 4859.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 13:00:00 | 4941.50 | 4951.24 | 4859.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 4886.95 | 4936.71 | 4902.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 13:45:00 | 4907.45 | 4936.71 | 4902.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 4856.05 | 4920.58 | 4897.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:00:00 | 4856.05 | 4920.58 | 4897.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 15:15:00 | 4860.00 | 4908.46 | 4894.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 4877.80 | 4908.46 | 4894.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 11:15:00 | 4846.95 | 4887.77 | 4887.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 4846.95 | 4887.77 | 4887.78 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 4955.15 | 4898.60 | 4892.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 5088.55 | 5009.44 | 4966.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 14:15:00 | 5070.85 | 5079.13 | 5023.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 5070.85 | 5079.13 | 5023.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 5060.55 | 5070.21 | 5045.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:30:00 | 5048.00 | 5070.21 | 5045.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 5033.05 | 5062.78 | 5044.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 4986.30 | 5062.78 | 5044.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 5034.90 | 5057.20 | 5043.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 5078.20 | 5057.20 | 5043.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 15:15:00 | 5000.00 | 5043.66 | 5044.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 5000.00 | 5043.66 | 5044.28 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 5077.95 | 5050.51 | 5047.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 5120.00 | 5064.41 | 5053.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 14:15:00 | 5299.45 | 5315.14 | 5258.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 15:00:00 | 5299.45 | 5315.14 | 5258.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 5250.00 | 5302.11 | 5257.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:30:00 | 5313.80 | 5287.48 | 5272.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 13:30:00 | 5347.00 | 5305.93 | 5283.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 14:15:00 | 5576.90 | 5601.27 | 5601.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 14:15:00 | 5576.90 | 5601.27 | 5601.52 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 09:15:00 | 5640.00 | 5605.61 | 5603.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 10:15:00 | 5981.40 | 5680.77 | 5637.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 14:15:00 | 5825.00 | 5859.47 | 5785.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 14:15:00 | 5825.00 | 5859.47 | 5785.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 5825.00 | 5859.47 | 5785.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 5825.00 | 5859.47 | 5785.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 5825.00 | 5874.31 | 5833.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 5825.00 | 5874.31 | 5833.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 5839.20 | 5867.29 | 5833.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:30:00 | 5878.05 | 5867.77 | 5837.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:30:00 | 5857.30 | 5864.07 | 5851.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:00:00 | 5859.85 | 5864.07 | 5851.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 12:00:00 | 5914.90 | 5876.01 | 5859.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 12:15:00 | 5873.10 | 5875.43 | 5860.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 12:30:00 | 5845.00 | 5875.43 | 5860.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 5854.25 | 5871.20 | 5860.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:00:00 | 5854.25 | 5871.20 | 5860.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 5850.90 | 5867.14 | 5859.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 5849.90 | 5867.14 | 5859.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 5837.95 | 5861.30 | 5857.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 5865.00 | 5861.30 | 5857.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 5805.00 | 5850.04 | 5852.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 5805.00 | 5850.04 | 5852.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 5783.00 | 5814.70 | 5831.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 5905.00 | 5808.73 | 5815.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 5905.00 | 5808.73 | 5815.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 5905.00 | 5808.73 | 5815.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 5905.00 | 5808.73 | 5815.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 5906.60 | 5828.31 | 5823.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 11:15:00 | 5922.10 | 5883.06 | 5859.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 12:15:00 | 5876.05 | 5881.65 | 5860.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 12:45:00 | 5877.35 | 5881.65 | 5860.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 5857.45 | 5875.59 | 5861.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:45:00 | 5859.95 | 5875.59 | 5861.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 15:15:00 | 5860.00 | 5872.47 | 5861.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 5885.95 | 5872.47 | 5861.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:30:00 | 5861.70 | 5868.94 | 5861.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 5840.00 | 5863.15 | 5859.74 | SL hit (close<static) qty=1.00 sl=5851.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 5892.35 | 5925.76 | 5926.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 09:15:00 | 5850.00 | 5881.24 | 5901.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 5965.20 | 5845.64 | 5863.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 5965.20 | 5845.64 | 5863.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 5965.20 | 5845.64 | 5863.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:15:00 | 5969.05 | 5845.64 | 5863.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 5893.45 | 5855.20 | 5866.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:15:00 | 5880.00 | 5855.20 | 5866.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:15:00 | 5876.95 | 5864.35 | 5865.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 11:15:00 | 5910.00 | 5874.38 | 5869.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 5910.00 | 5874.38 | 5869.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 13:15:00 | 5994.80 | 5907.20 | 5886.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 11:15:00 | 6200.10 | 6261.17 | 6176.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 12:00:00 | 6200.10 | 6261.17 | 6176.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 6199.20 | 6237.33 | 6179.77 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 11:15:00 | 6080.00 | 6154.43 | 6156.34 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 6243.85 | 6166.39 | 6158.87 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 6119.20 | 6150.59 | 6153.98 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 09:15:00 | 6221.70 | 6156.32 | 6154.94 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 13:15:00 | 6160.80 | 6179.84 | 6179.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 14:15:00 | 6125.90 | 6169.05 | 6174.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-11 10:15:00 | 6211.05 | 6175.23 | 6176.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 10:15:00 | 6211.05 | 6175.23 | 6176.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 6211.05 | 6175.23 | 6176.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 6211.05 | 6175.23 | 6176.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 6163.45 | 6172.87 | 6174.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 12:45:00 | 6138.30 | 6166.02 | 6171.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 6150.40 | 6146.02 | 6158.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 6239.95 | 6164.81 | 6166.08 | SL hit (close>static) qty=1.00 sl=6217.95 alert=retest2 |

### Cycle 114 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 6191.10 | 6170.07 | 6168.36 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 14:15:00 | 6106.00 | 6159.17 | 6164.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 15:15:00 | 6095.00 | 6146.34 | 6157.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 11:15:00 | 5985.00 | 5982.10 | 6032.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-17 12:00:00 | 5985.00 | 5982.10 | 6032.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 5417.45 | 5632.77 | 5728.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 15:00:00 | 5417.45 | 5632.77 | 5728.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 5447.00 | 5425.35 | 5456.29 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 5866.65 | 5527.89 | 5491.42 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 5521.10 | 5593.15 | 5596.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 5488.25 | 5572.17 | 5586.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 5642.30 | 5540.88 | 5559.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 5642.30 | 5540.88 | 5559.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 5642.30 | 5540.88 | 5559.77 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 11:15:00 | 5650.05 | 5584.97 | 5577.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 10:15:00 | 5706.75 | 5650.68 | 5623.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 12:15:00 | 5646.25 | 5650.80 | 5628.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-08 12:45:00 | 5645.45 | 5650.80 | 5628.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 5620.00 | 5642.91 | 5628.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 5620.00 | 5642.91 | 5628.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 5617.00 | 5637.73 | 5627.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 5692.55 | 5637.73 | 5627.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 09:15:00 | 5690.00 | 5772.08 | 5776.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 09:15:00 | 5690.00 | 5772.08 | 5776.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 10:15:00 | 5679.00 | 5753.46 | 5767.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 09:15:00 | 5822.80 | 5728.69 | 5742.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 09:15:00 | 5822.80 | 5728.69 | 5742.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 5822.80 | 5728.69 | 5742.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:30:00 | 5831.20 | 5728.69 | 5742.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 5805.80 | 5744.11 | 5748.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:30:00 | 5820.00 | 5744.11 | 5748.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 11:15:00 | 5795.15 | 5754.32 | 5752.35 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 5694.45 | 5744.46 | 5750.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 11:15:00 | 5642.90 | 5724.15 | 5740.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 15:15:00 | 5690.00 | 5688.72 | 5715.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 09:15:00 | 5678.00 | 5688.72 | 5715.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 5625.20 | 5676.01 | 5707.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 5593.90 | 5633.89 | 5672.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 15:15:00 | 5590.00 | 5633.89 | 5672.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 15:15:00 | 5314.20 | 5409.30 | 5468.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 15:15:00 | 5310.50 | 5409.30 | 5468.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 13:15:00 | 5315.00 | 5284.68 | 5338.83 | SL hit (close>ema200) qty=0.50 sl=5284.68 alert=retest2 |

### Cycle 122 — BUY (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 09:15:00 | 5203.80 | 5145.22 | 5138.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 5311.00 | 5235.32 | 5194.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-08 09:15:00 | 5316.90 | 5327.97 | 5292.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:30:00 | 5305.20 | 5327.97 | 5292.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 5277.25 | 5313.67 | 5292.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 5277.25 | 5313.67 | 5292.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 5279.95 | 5306.92 | 5291.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 13:45:00 | 5295.40 | 5304.68 | 5291.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:45:00 | 5306.00 | 5303.44 | 5292.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 10:15:00 | 5257.10 | 5285.38 | 5285.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 5257.10 | 5285.38 | 5285.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 5211.40 | 5270.59 | 5279.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 5348.35 | 5256.52 | 5264.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 5348.35 | 5256.52 | 5264.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 5348.35 | 5256.52 | 5264.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 5351.65 | 5256.52 | 5264.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 5355.95 | 5276.40 | 5273.24 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 5146.95 | 5261.83 | 5270.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 5121.25 | 5178.77 | 5198.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 5210.15 | 5157.16 | 5173.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-22 09:15:00 | 5210.15 | 5157.16 | 5173.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 5210.15 | 5157.16 | 5173.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:15:00 | 5240.00 | 5157.16 | 5173.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 5249.00 | 5175.53 | 5180.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:30:00 | 5231.00 | 5175.53 | 5180.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 5251.35 | 5190.70 | 5186.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5289.95 | 5229.75 | 5209.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 13:15:00 | 5249.95 | 5251.59 | 5228.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 13:30:00 | 5260.75 | 5251.59 | 5228.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 5255.70 | 5252.41 | 5230.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:30:00 | 5255.75 | 5252.41 | 5230.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 5215.20 | 5244.58 | 5230.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 5215.20 | 5244.58 | 5230.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 5289.20 | 5253.51 | 5236.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 5296.70 | 5262.59 | 5247.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:00:00 | 5301.90 | 5271.75 | 5254.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:30:00 | 5297.80 | 5279.49 | 5261.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 5295.10 | 5314.44 | 5305.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 5313.35 | 5314.22 | 5306.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:00:00 | 5313.35 | 5314.22 | 5306.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 5317.75 | 5314.93 | 5307.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:45:00 | 5316.05 | 5314.93 | 5307.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 5310.50 | 5352.84 | 5341.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 5310.70 | 5352.84 | 5341.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 5367.75 | 5355.82 | 5343.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 5317.40 | 5355.82 | 5343.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 5309.05 | 5349.64 | 5346.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 5313.00 | 5349.64 | 5346.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 5292.60 | 5338.23 | 5341.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 5292.60 | 5338.23 | 5341.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 5192.00 | 5283.99 | 5310.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 5109.30 | 5088.22 | 5138.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 10:00:00 | 5109.30 | 5088.22 | 5138.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 5072.30 | 5086.70 | 5114.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 09:15:00 | 5056.25 | 5085.12 | 5100.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 12:15:00 | 5162.30 | 5036.41 | 5029.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 12:15:00 | 5162.30 | 5036.41 | 5029.38 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 12:15:00 | 5013.25 | 5029.34 | 5031.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 4962.00 | 5010.18 | 5021.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 14:15:00 | 4838.45 | 4786.61 | 4855.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 15:15:00 | 4850.00 | 4786.61 | 4855.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 4850.00 | 4799.28 | 4854.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 09:15:00 | 4799.90 | 4799.28 | 4854.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:45:00 | 4816.00 | 4810.70 | 4847.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 13:15:00 | 4867.70 | 4824.06 | 4847.12 | SL hit (close>static) qty=1.00 sl=4865.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 4906.60 | 4849.80 | 4842.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 11:15:00 | 5104.40 | 4918.49 | 4883.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 5225.85 | 5237.09 | 5140.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 11:00:00 | 5293.90 | 5248.45 | 5154.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 13:15:00 | 5295.90 | 5262.01 | 5177.39 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 14:00:00 | 5290.70 | 5267.75 | 5187.69 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 15:15:00 | 5300.50 | 5267.90 | 5195.04 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 5329.35 | 5279.67 | 5236.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-03 12:15:00 | 5237.45 | 5269.19 | 5248.66 | SL hit (close<ema400) qty=1.00 sl=5248.66 alert=retest1 |

### Cycle 131 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 5134.25 | 5219.36 | 5230.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 5114.90 | 5170.76 | 5201.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 5206.90 | 5173.54 | 5196.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 5206.90 | 5173.54 | 5196.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 5206.90 | 5173.54 | 5196.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 5225.45 | 5173.54 | 5196.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 5340.00 | 5206.84 | 5209.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 5340.00 | 5206.84 | 5209.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 11:15:00 | 5431.75 | 5251.82 | 5230.07 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 5309.05 | 5334.11 | 5335.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 5230.35 | 5300.82 | 5318.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 13:15:00 | 5091.45 | 5089.25 | 5158.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 14:00:00 | 5091.45 | 5089.25 | 5158.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 4909.35 | 4949.78 | 4979.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 09:15:00 | 4906.85 | 4937.97 | 4948.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 4661.51 | 4704.86 | 4762.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-28 11:15:00 | 4416.17 | 4541.85 | 4629.52 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 134 — BUY (started 2025-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 13:15:00 | 4538.00 | 4531.38 | 4530.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 10:15:00 | 4560.25 | 4540.56 | 4535.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 15:15:00 | 4535.00 | 4556.06 | 4546.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 15:15:00 | 4535.00 | 4556.06 | 4546.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 15:15:00 | 4535.00 | 4556.06 | 4546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:15:00 | 4454.90 | 4556.06 | 4546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 4402.80 | 4525.41 | 4533.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 10:15:00 | 4359.65 | 4422.12 | 4459.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 15:15:00 | 4410.00 | 4392.45 | 4427.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-06 09:15:00 | 4402.00 | 4392.45 | 4427.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 4397.20 | 4393.40 | 4424.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 14:00:00 | 4375.25 | 4393.70 | 4415.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 4368.30 | 4401.91 | 4415.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 4156.49 | 4240.03 | 4300.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 4149.89 | 4240.03 | 4300.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 4121.90 | 4117.09 | 4178.05 | SL hit (close>ema200) qty=0.50 sl=4117.09 alert=retest2 |

### Cycle 136 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 4258.50 | 4186.68 | 4183.60 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 4124.00 | 4183.11 | 4184.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 4082.35 | 4162.95 | 4175.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 14:15:00 | 4130.75 | 4123.95 | 4149.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 14:15:00 | 4130.75 | 4123.95 | 4149.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 4130.75 | 4123.95 | 4149.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:30:00 | 4166.70 | 4123.95 | 4149.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 4225.00 | 4144.16 | 4156.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:15:00 | 4115.80 | 4144.16 | 4156.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 09:45:00 | 4099.90 | 4134.08 | 4150.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 4090.00 | 4139.22 | 4145.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 4158.90 | 4131.79 | 4131.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 4158.90 | 4131.79 | 4131.17 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 4110.00 | 4127.43 | 4129.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 4109.30 | 4123.80 | 4127.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-19 14:15:00 | 4135.40 | 4126.12 | 4128.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 14:15:00 | 4135.40 | 4126.12 | 4128.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 4135.40 | 4126.12 | 4128.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 4135.40 | 4126.12 | 4128.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 4128.05 | 4126.51 | 4128.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:15:00 | 4079.10 | 4126.51 | 4128.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 4071.20 | 4115.45 | 4122.97 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 09:15:00 | 4367.35 | 4155.58 | 4132.25 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 13:15:00 | 4140.20 | 4174.07 | 4175.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 09:15:00 | 4051.00 | 4138.56 | 4158.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 13:15:00 | 4012.75 | 4000.20 | 4050.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 14:00:00 | 4012.75 | 4000.20 | 4050.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 4042.20 | 4008.60 | 4049.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 4042.20 | 4008.60 | 4049.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 4082.00 | 4023.28 | 4052.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 3958.15 | 4023.28 | 4052.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 3933.45 | 4005.31 | 4041.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:30:00 | 3922.20 | 3979.09 | 4022.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 12:00:00 | 3916.65 | 3979.09 | 4022.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 3905.05 | 3951.04 | 3993.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 4034.00 | 3986.99 | 3985.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 4034.00 | 3986.99 | 3985.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 4045.00 | 4006.59 | 3994.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 4119.60 | 4125.86 | 4099.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 4119.60 | 4125.86 | 4099.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 4119.60 | 4125.86 | 4099.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 4102.30 | 4125.86 | 4099.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 4038.50 | 4126.75 | 4116.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 4041.30 | 4126.75 | 4116.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 4028.65 | 4107.13 | 4108.58 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 4112.05 | 4079.38 | 4079.38 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 13:15:00 | 4080.00 | 4083.93 | 4084.31 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 4094.00 | 4084.35 | 4084.24 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 13:15:00 | 4063.20 | 4084.94 | 4085.44 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 4113.50 | 4086.42 | 4085.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 4151.60 | 4124.87 | 4113.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 11:15:00 | 4096.15 | 4122.60 | 4114.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 11:15:00 | 4096.15 | 4122.60 | 4114.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 4096.15 | 4122.60 | 4114.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:00:00 | 4096.15 | 4122.60 | 4114.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 12:15:00 | 4094.65 | 4117.01 | 4112.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 12:45:00 | 4098.65 | 4117.01 | 4112.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 4101.00 | 4109.30 | 4109.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 4069.90 | 4091.99 | 4099.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 4005.60 | 3994.65 | 4021.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 11:00:00 | 4005.60 | 3994.65 | 4021.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 4059.15 | 4007.55 | 4024.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:45:00 | 4070.70 | 4007.55 | 4024.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 4031.00 | 4012.24 | 4025.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:15:00 | 4010.00 | 4012.24 | 4025.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 3999.90 | 4009.79 | 4023.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:00:00 | 3993.00 | 4006.43 | 4020.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 13:15:00 | 4043.80 | 4025.88 | 4024.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 13:15:00 | 4043.80 | 4025.88 | 4024.65 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 3991.45 | 4022.91 | 4023.93 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 4048.90 | 4027.29 | 4025.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 4078.00 | 4038.82 | 4030.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 11:15:00 | 4026.50 | 4036.98 | 4031.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 11:15:00 | 4026.50 | 4036.98 | 4031.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 4026.50 | 4036.98 | 4031.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 4026.50 | 4036.98 | 4031.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 4031.95 | 4035.97 | 4031.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:30:00 | 4031.40 | 4035.97 | 4031.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 4043.35 | 4037.45 | 4032.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 4029.70 | 4037.45 | 4032.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 4055.15 | 4047.77 | 4038.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 4030.25 | 4047.77 | 4038.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 3978.30 | 4033.88 | 4033.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 3978.30 | 4033.88 | 4033.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 10:15:00 | 3989.80 | 4025.06 | 4029.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 3828.80 | 3978.52 | 4003.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 3941.05 | 3920.75 | 3957.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 3944.45 | 3920.75 | 3957.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 3939.75 | 3924.55 | 3955.66 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 14:15:00 | 4019.90 | 3973.29 | 3969.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 15:15:00 | 4032.50 | 3985.14 | 3975.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 3967.40 | 3981.59 | 3974.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 3967.40 | 3981.59 | 3974.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 3967.40 | 3981.59 | 3974.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 11:30:00 | 4003.10 | 3987.46 | 3978.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 4002.85 | 3987.46 | 3978.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 10:15:00 | 4216.30 | 4235.83 | 4236.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 4216.30 | 4235.83 | 4236.99 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 4272.00 | 4241.49 | 4238.95 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 14:15:00 | 4260.10 | 4277.48 | 4278.50 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 4290.40 | 4278.98 | 4278.77 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 4266.10 | 4277.28 | 4278.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 4244.60 | 4270.74 | 4275.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 4256.50 | 4245.50 | 4259.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 4256.50 | 4245.50 | 4259.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 4256.50 | 4245.50 | 4259.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 4256.50 | 4245.50 | 4259.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 4248.10 | 4246.02 | 4258.48 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 4266.40 | 4263.95 | 4263.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 4295.50 | 4270.26 | 4266.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 4241.00 | 4268.77 | 4267.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 15:15:00 | 4241.00 | 4268.77 | 4267.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 4241.00 | 4268.77 | 4267.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 4171.00 | 4268.77 | 4267.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 4194.00 | 4253.82 | 4260.52 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 4258.50 | 4252.41 | 4252.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 4290.10 | 4261.03 | 4256.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 4430.10 | 4436.36 | 4393.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 09:30:00 | 4431.60 | 4436.36 | 4393.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 4420.00 | 4428.28 | 4405.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:30:00 | 4411.00 | 4428.28 | 4405.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 4462.10 | 4432.12 | 4411.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 4427.80 | 4432.12 | 4411.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 5742.00 | 5776.04 | 5744.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 5742.00 | 5776.04 | 5744.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 5730.50 | 5766.93 | 5743.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 5730.50 | 5766.93 | 5743.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 5708.00 | 5755.14 | 5740.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:00:00 | 5708.00 | 5755.14 | 5740.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 5694.50 | 5743.02 | 5736.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:00:00 | 5694.50 | 5743.02 | 5736.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 5704.50 | 5728.19 | 5730.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 09:15:00 | 5660.00 | 5714.55 | 5724.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5741.00 | 5671.82 | 5690.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 5741.00 | 5671.82 | 5690.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 5764.00 | 5690.25 | 5697.18 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 5773.50 | 5706.90 | 5704.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 5780.00 | 5750.54 | 5730.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 15:15:00 | 5753.00 | 5766.01 | 5747.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 09:15:00 | 5870.00 | 5766.01 | 5747.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5874.00 | 5856.65 | 5816.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5900.50 | 5856.65 | 5816.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:00:00 | 5887.50 | 5870.99 | 5836.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 5889.00 | 5870.79 | 5839.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 15:15:00 | 5889.00 | 5870.79 | 5839.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5843.00 | 5868.14 | 5844.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 5843.00 | 5868.14 | 5844.18 | SL hit (close<ema400) qty=1.00 sl=5844.18 alert=retest1 |

### Cycle 165 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 5803.00 | 5848.49 | 5854.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 15:15:00 | 5774.50 | 5817.02 | 5836.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5779.00 | 5726.38 | 5768.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 5760.00 | 5726.38 | 5768.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 5742.50 | 5729.60 | 5766.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 5694.50 | 5724.28 | 5760.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 12:15:00 | 5650.00 | 5613.65 | 5610.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 12:15:00 | 5650.00 | 5613.65 | 5610.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 5670.50 | 5625.02 | 5616.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 5647.50 | 5660.41 | 5641.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:45:00 | 5640.00 | 5660.41 | 5641.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 5656.50 | 5659.63 | 5643.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:45:00 | 5646.00 | 5659.63 | 5643.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 5704.00 | 5697.34 | 5686.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:45:00 | 5777.50 | 5725.70 | 5708.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5669.50 | 5807.17 | 5801.68 | SL hit (close<static) qty=1.00 sl=5685.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 5650.00 | 5775.74 | 5787.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 11:15:00 | 5649.00 | 5750.39 | 5775.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 5521.00 | 5518.34 | 5576.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 14:45:00 | 5529.50 | 5518.34 | 5576.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 5335.50 | 5363.74 | 5391.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 5318.50 | 5352.23 | 5381.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 11:15:00 | 5325.50 | 5336.14 | 5356.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 5331.00 | 5340.99 | 5354.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 5330.00 | 5339.40 | 5352.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 5267.50 | 5323.51 | 5342.59 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 5407.50 | 5309.97 | 5306.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 09:15:00 | 5407.50 | 5309.97 | 5306.68 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 5305.00 | 5330.79 | 5331.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 15:15:00 | 5265.00 | 5317.63 | 5325.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 5281.50 | 5276.54 | 5297.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 5281.50 | 5276.54 | 5297.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5292.50 | 5278.66 | 5292.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:00:00 | 5292.50 | 5278.66 | 5292.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 5289.00 | 5280.73 | 5292.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 5285.00 | 5280.73 | 5292.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 5124.00 | 5198.81 | 5233.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 5096.50 | 5166.48 | 5211.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 5098.00 | 5024.04 | 5016.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 5098.00 | 5024.04 | 5016.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 09:15:00 | 5430.50 | 5165.20 | 5114.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 15:15:00 | 5650.00 | 5668.39 | 5567.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:15:00 | 5660.00 | 5668.39 | 5567.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 5688.50 | 5688.79 | 5652.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 5730.00 | 5699.43 | 5660.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 5641.50 | 5685.83 | 5671.62 | SL hit (close<static) qty=1.00 sl=5644.50 alert=retest2 |

### Cycle 171 — SELL (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 09:15:00 | 5565.00 | 5661.66 | 5661.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 11:15:00 | 5522.50 | 5617.16 | 5640.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 5144.50 | 5144.35 | 5185.72 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-02 11:15:00 | 5132.50 | 5144.35 | 5185.72 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 5149.50 | 5127.94 | 5157.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 5143.00 | 5127.94 | 5157.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 5187.00 | 5141.36 | 5158.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 5187.00 | 5141.36 | 5158.48 | SL hit (close>ema400) qty=1.00 sl=5158.48 alert=retest1 |

### Cycle 172 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 5214.50 | 5171.67 | 5168.23 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 5141.00 | 5166.47 | 5168.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 5097.50 | 5140.08 | 5155.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 5153.00 | 5134.52 | 5146.69 | EMA400 retest candle locked (from downside) |

### Cycle 174 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 5173.50 | 5148.92 | 5146.51 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 5143.00 | 5145.46 | 5145.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 10:15:00 | 5126.00 | 5138.59 | 5142.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 5139.50 | 5136.33 | 5140.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 5139.50 | 5136.33 | 5140.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 5162.50 | 5141.56 | 5142.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 5163.00 | 5141.56 | 5142.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 5167.00 | 5146.65 | 5144.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 09:15:00 | 5210.50 | 5159.42 | 5150.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 5197.00 | 5201.16 | 5180.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 09:30:00 | 5182.00 | 5201.16 | 5180.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 5190.00 | 5198.93 | 5181.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 5186.00 | 5198.93 | 5181.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5189.50 | 5197.04 | 5182.36 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 10:15:00 | 5164.50 | 5174.40 | 5175.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 11:15:00 | 5149.50 | 5169.42 | 5173.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 5174.00 | 5157.58 | 5164.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 5171.00 | 5157.58 | 5164.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 5184.00 | 5162.87 | 5166.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 5184.00 | 5162.87 | 5166.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 5196.50 | 5169.59 | 5168.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 5225.00 | 5191.79 | 5180.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 5193.00 | 5198.65 | 5189.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 5121.50 | 5188.72 | 5186.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 10:15:00 | 5146.50 | 5180.27 | 5182.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 5138.00 | 5165.54 | 5175.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 5139.50 | 5138.68 | 5154.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:00:00 | 5139.50 | 5138.68 | 5154.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4958.00 | 4954.18 | 4980.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 4980.50 | 4954.18 | 4980.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 4979.00 | 4960.63 | 4978.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 4979.00 | 4960.63 | 4978.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 4977.00 | 4963.91 | 4978.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 4977.00 | 4963.91 | 4978.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 4990.50 | 4969.23 | 4979.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 4990.50 | 4969.23 | 4979.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 4878.50 | 4951.08 | 4970.52 | EMA400 retest candle locked (from downside) |

### Cycle 180 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 5014.50 | 4975.81 | 4975.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 14:15:00 | 5032.50 | 4993.34 | 4983.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5220.00 | 5223.46 | 5188.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 5240.00 | 5223.46 | 5188.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:00:00 | 5236.00 | 5226.80 | 5201.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 5237.50 | 5225.54 | 5202.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:30:00 | 5244.50 | 5229.32 | 5208.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 5185.00 | 5222.76 | 5213.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 5185.00 | 5222.76 | 5213.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 5161.00 | 5210.41 | 5209.13 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 5161.00 | 5210.41 | 5209.13 | SL hit (close<static) qty=1.00 sl=5175.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 14:15:00 | 5329.50 | 5379.60 | 5382.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 15:15:00 | 5320.50 | 5367.78 | 5376.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 5372.50 | 5326.62 | 5341.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 5372.50 | 5326.62 | 5341.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 5359.00 | 5333.10 | 5343.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 5325.00 | 5332.78 | 5341.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:00:00 | 5328.00 | 5331.82 | 5340.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 14:45:00 | 5322.50 | 5330.27 | 5335.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 5310.00 | 5328.23 | 5333.75 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5330.50 | 5328.69 | 5333.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 5426.00 | 5348.15 | 5341.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 5426.00 | 5348.15 | 5341.87 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 5284.00 | 5331.72 | 5337.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 09:15:00 | 5252.00 | 5297.07 | 5310.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 12:15:00 | 5279.50 | 5277.71 | 5296.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 5279.50 | 5277.71 | 5296.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 5297.50 | 5280.84 | 5294.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 11:00:00 | 5270.00 | 5278.22 | 5290.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:00:00 | 5269.50 | 5274.49 | 5285.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 5264.50 | 5272.49 | 5283.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 5325.50 | 5259.75 | 5268.32 | SL hit (close>static) qty=1.00 sl=5324.50 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5118.50 | 5101.17 | 5100.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 14:15:00 | 5138.50 | 5108.64 | 5104.39 | Break + close above crossover candle high |

### Cycle 185 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 5064.50 | 5100.27 | 5101.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 5024.50 | 5085.11 | 5094.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 5103.50 | 5041.18 | 5061.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 5103.50 | 5041.18 | 5061.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 4999.00 | 5032.74 | 5055.88 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 10:15:00 | 5100.00 | 5053.08 | 5051.78 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 5048.00 | 5062.23 | 5063.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5010.50 | 5050.38 | 5057.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 11:15:00 | 5035.00 | 5032.50 | 5045.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 12:00:00 | 5035.00 | 5032.50 | 5045.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 5049.00 | 5035.80 | 5045.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 5049.00 | 5035.80 | 5045.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 5028.00 | 5034.24 | 5043.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:15:00 | 5020.50 | 5034.24 | 5043.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:00:00 | 5025.00 | 5018.82 | 5027.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 10:15:00 | 5029.00 | 4998.60 | 4996.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 5029.00 | 4998.60 | 4996.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 5050.50 | 5008.98 | 5001.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 5005.00 | 5010.75 | 5003.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:00:00 | 5005.00 | 5010.75 | 5003.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4997.50 | 5008.10 | 5003.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 14:45:00 | 4995.00 | 5008.10 | 5003.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4973.00 | 5001.08 | 5000.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5048.00 | 5001.08 | 5000.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4982.00 | 4997.26 | 4998.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 189 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 4982.00 | 4997.26 | 4998.82 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 13:15:00 | 5035.00 | 5003.51 | 5000.90 | EMA200 above EMA400 |

### Cycle 191 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 4995.00 | 4998.69 | 4998.97 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 5022.50 | 5003.45 | 5001.11 | EMA200 above EMA400 |

### Cycle 193 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 4990.00 | 4998.89 | 4999.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 4975.50 | 4993.03 | 4996.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 4987.70 | 4973.61 | 4981.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 5023.60 | 4973.61 | 4981.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 4995.10 | 4977.91 | 4982.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 4995.10 | 4977.91 | 4982.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 4967.40 | 4975.81 | 4981.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:00:00 | 4963.00 | 4973.25 | 4979.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 4958.00 | 4976.75 | 4979.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 4965.70 | 4960.92 | 4967.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 4999.60 | 4973.28 | 4972.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 4999.60 | 4973.28 | 4972.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 15:15:00 | 5005.00 | 4987.32 | 4980.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 4995.00 | 4996.14 | 4988.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 4951.00 | 4996.14 | 4988.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 4957.00 | 4988.31 | 4985.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:30:00 | 4974.70 | 4988.31 | 4985.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 4938.90 | 4978.43 | 4981.71 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 11:15:00 | 4999.00 | 4980.26 | 4978.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 5001.80 | 4987.76 | 4983.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 5017.00 | 5019.41 | 5004.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-11 09:15:00 | 5011.10 | 5019.41 | 5004.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 5018.20 | 5019.17 | 5005.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:45:00 | 5005.90 | 5019.17 | 5005.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 5011.50 | 5017.63 | 5006.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 5011.50 | 5017.63 | 5006.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5005.50 | 5015.21 | 5006.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:30:00 | 5019.80 | 5015.21 | 5006.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5007.40 | 5013.65 | 5006.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 5007.40 | 5013.65 | 5006.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 5010.00 | 5012.92 | 5006.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:45:00 | 5001.00 | 5012.92 | 5006.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 5006.90 | 5011.71 | 5006.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 14:45:00 | 5009.80 | 5011.71 | 5006.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 5016.00 | 5012.57 | 5007.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 4991.00 | 5012.57 | 5007.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 5008.70 | 5011.80 | 5007.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:15:00 | 4996.00 | 5011.80 | 5007.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 4978.30 | 5005.10 | 5004.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 4978.30 | 5005.10 | 5004.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 11:15:00 | 4980.10 | 5000.10 | 5002.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 4976.10 | 4992.26 | 4998.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 5018.00 | 4997.41 | 5000.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 5018.00 | 4997.41 | 5000.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 5034.00 | 5004.73 | 5003.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 15:15:00 | 5037.00 | 5022.52 | 5016.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 5022.00 | 5030.43 | 5023.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 12:45:00 | 5021.30 | 5030.43 | 5023.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 5005.10 | 5025.36 | 5021.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 5005.10 | 5025.36 | 5021.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5055.90 | 5031.47 | 5024.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 10:30:00 | 5090.00 | 5051.41 | 5036.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 5087.00 | 5089.77 | 5078.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:30:00 | 5083.90 | 5086.95 | 5078.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 13:15:00 | 5083.70 | 5082.01 | 5077.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 5129.90 | 5092.02 | 5082.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:15:00 | 5061.40 | 5092.02 | 5082.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 5053.90 | 5084.40 | 5080.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:30:00 | 5054.00 | 5084.40 | 5080.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 5029.50 | 5073.42 | 5075.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 5029.50 | 5073.42 | 5075.70 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 5116.00 | 5070.65 | 5070.39 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 5037.60 | 5070.24 | 5071.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 5030.80 | 5054.94 | 5063.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 5055.60 | 5051.21 | 5058.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:45:00 | 5069.80 | 5051.21 | 5058.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 5052.50 | 5051.47 | 5058.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 5064.80 | 5051.47 | 5058.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 5059.90 | 5053.15 | 5058.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 5028.40 | 5053.15 | 5058.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5017.50 | 5046.02 | 5054.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:30:00 | 5011.20 | 5036.70 | 5049.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 4999.40 | 5036.70 | 5049.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 4952.40 | 4902.28 | 4898.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 4952.40 | 4902.28 | 4898.98 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 4858.80 | 4907.40 | 4909.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 4857.90 | 4889.73 | 4900.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 11:15:00 | 4857.00 | 4849.45 | 4871.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:30:00 | 4854.80 | 4849.45 | 4871.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 4859.20 | 4852.15 | 4868.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:45:00 | 4860.50 | 4852.15 | 4868.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 4856.50 | 4853.02 | 4867.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 4883.20 | 4853.02 | 4867.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 4868.90 | 4856.19 | 4867.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 4800.00 | 4856.19 | 4867.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 4840.00 | 4843.34 | 4843.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 4598.00 | 4659.85 | 4698.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 4776.60 | 4652.02 | 4669.49 | SL hit (close>ema200) qty=0.50 sl=4652.02 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 4727.40 | 4687.40 | 4683.12 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 4640.00 | 4684.62 | 4687.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 4626.70 | 4673.03 | 4681.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 4624.00 | 4615.69 | 4636.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 4651.10 | 4615.69 | 4636.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4667.20 | 4626.00 | 4639.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 4689.60 | 4626.00 | 4639.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4660.70 | 4632.94 | 4641.06 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 14:15:00 | 4674.00 | 4650.28 | 4647.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 4745.00 | 4669.22 | 4656.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 4606.00 | 4656.58 | 4651.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 4606.00 | 4656.58 | 4651.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 4604.40 | 4646.14 | 4647.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 13:15:00 | 4590.80 | 4624.58 | 4636.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 4622.50 | 4588.29 | 4603.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 4622.50 | 4588.29 | 4603.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4575.70 | 4585.77 | 4600.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 4552.30 | 4585.77 | 4600.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 4634.90 | 4574.83 | 4566.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 4634.90 | 4574.83 | 4566.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 4650.00 | 4589.87 | 4574.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 12:15:00 | 4700.00 | 4703.44 | 4671.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:30:00 | 4688.30 | 4703.44 | 4671.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4642.00 | 4688.48 | 4674.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 4640.80 | 4688.48 | 4674.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 4616.00 | 4673.99 | 4668.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 4616.00 | 4673.99 | 4668.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 209 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 4645.50 | 4661.70 | 4663.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 13:15:00 | 4594.50 | 4648.26 | 4657.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4666.50 | 4636.32 | 4648.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 4666.50 | 4636.32 | 4648.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 4683.80 | 4645.82 | 4651.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:00:00 | 4683.80 | 4645.82 | 4651.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 4693.20 | 4659.80 | 4657.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 4719.50 | 4671.74 | 4662.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 5120.80 | 5122.99 | 5026.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:15:00 | 5123.70 | 5122.99 | 5026.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5034.50 | 5106.57 | 5055.69 | EMA400 retest candle locked (from upside) |

### Cycle 211 — SELL (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 11:15:00 | 5018.90 | 5041.43 | 5043.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 13:15:00 | 5001.20 | 5029.65 | 5037.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 5026.10 | 5020.54 | 5029.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:30:00 | 5022.00 | 5020.54 | 5029.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 5060.20 | 5028.47 | 5032.07 | EMA400 retest candle locked (from downside) |

### Cycle 212 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 5070.00 | 5036.78 | 5035.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 5119.90 | 5054.07 | 5043.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 5095.00 | 5114.61 | 5086.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 5088.80 | 5114.61 | 5086.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 5077.00 | 5107.09 | 5085.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 5079.30 | 5107.09 | 5085.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 5062.10 | 5098.09 | 5083.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 5061.30 | 5098.09 | 5083.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 5020.50 | 5072.38 | 5073.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 4990.00 | 5055.91 | 5066.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 5024.20 | 4981.88 | 5002.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 5024.20 | 4981.88 | 5002.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 4985.00 | 4982.50 | 5001.31 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-02-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 15:15:00 | 5066.60 | 5014.01 | 5013.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 09:15:00 | 5105.50 | 5032.31 | 5021.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 5100.60 | 5125.42 | 5099.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 5100.60 | 5125.42 | 5099.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 5092.70 | 5118.87 | 5099.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 5086.20 | 5118.87 | 5099.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 5080.20 | 5111.14 | 5097.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:30:00 | 5073.70 | 5111.14 | 5097.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 5082.60 | 5101.33 | 5094.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:30:00 | 5086.00 | 5101.33 | 5094.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 15:15:00 | 5085.00 | 5098.06 | 5094.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:15:00 | 5096.90 | 5098.06 | 5094.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 5121.80 | 5102.81 | 5096.59 | EMA400 retest candle locked (from upside) |

### Cycle 215 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 5068.00 | 5093.37 | 5094.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 15:15:00 | 5055.00 | 5085.69 | 5090.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 4877.50 | 4872.34 | 4928.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 4881.00 | 4872.34 | 4928.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 4879.00 | 4837.40 | 4878.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 4870.50 | 4837.40 | 4878.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 4853.50 | 4840.62 | 4875.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 4860.50 | 4840.62 | 4875.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 4868.50 | 4850.18 | 4874.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 4880.00 | 4850.18 | 4874.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 4863.00 | 4852.74 | 4873.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 4848.00 | 4851.19 | 4870.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 4605.60 | 4672.65 | 4712.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 4699.00 | 4655.01 | 4683.26 | SL hit (close>ema200) qty=0.50 sl=4655.01 alert=retest2 |

### Cycle 216 — BUY (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 09:15:00 | 4730.00 | 4691.59 | 4687.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 4785.00 | 4727.74 | 4709.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4724.50 | 4780.37 | 4753.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:30:00 | 4711.50 | 4780.37 | 4753.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 4764.00 | 4777.10 | 4754.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:30:00 | 4723.50 | 4777.10 | 4754.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 4733.50 | 4768.38 | 4752.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 4733.50 | 4768.38 | 4752.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 4722.00 | 4759.10 | 4749.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 4722.00 | 4759.10 | 4749.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 217 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 4720.00 | 4741.06 | 4742.91 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 4760.50 | 4744.95 | 4744.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 11:15:00 | 4796.00 | 4755.97 | 4749.64 | Break + close above crossover candle high |

### Cycle 219 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 4652.50 | 4743.56 | 4747.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 4625.50 | 4719.94 | 4736.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4647.50 | 4625.58 | 4672.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:45:00 | 4682.50 | 4625.58 | 4672.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 4672.50 | 4639.13 | 4667.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 4687.00 | 4639.13 | 4667.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 4673.00 | 4645.91 | 4668.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 4676.50 | 4645.91 | 4668.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 4671.50 | 4651.02 | 4668.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 15:00:00 | 4671.50 | 4651.02 | 4668.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 4674.50 | 4655.72 | 4668.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 4740.50 | 4655.72 | 4668.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 4759.00 | 4676.38 | 4677.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 4746.50 | 4676.38 | 4677.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 220 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4741.00 | 4689.30 | 4682.89 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4649.50 | 4699.19 | 4699.87 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 12:15:00 | 4724.50 | 4700.13 | 4699.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-30 13:15:00 | 4732.50 | 4706.60 | 4702.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 4764.50 | 4797.76 | 4765.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 4764.50 | 4797.76 | 4765.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 4782.40 | 4794.69 | 4767.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:00:00 | 4807.60 | 4792.68 | 4770.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 4804.90 | 4831.93 | 4800.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 4804.40 | 4824.38 | 4811.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 4773.10 | 4801.65 | 4803.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 223 — SELL (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 12:15:00 | 4773.10 | 4801.65 | 4803.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 13:15:00 | 4770.10 | 4795.34 | 4800.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 4841.80 | 4799.62 | 4800.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:45:00 | 4850.00 | 4799.62 | 4800.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 4851.60 | 4810.02 | 4805.06 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 13:15:00 | 4739.80 | 4798.90 | 4806.09 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 12:15:00 | 4825.00 | 4807.35 | 4805.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 4852.70 | 4816.42 | 4809.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4820.30 | 4831.07 | 4819.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4843.60 | 4834.08 | 4821.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 4886.30 | 4906.25 | 4908.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 4886.30 | 4906.25 | 4908.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 4855.70 | 4896.14 | 4904.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4822.60 | 4807.54 | 4838.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 10:00:00 | 4822.60 | 4807.54 | 4838.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 4708.50 | 4772.98 | 4804.66 | EMA400 retest candle locked (from downside) |

### Cycle 228 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 4807.20 | 4776.56 | 4773.74 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 13:15:00 | 4771.10 | 4787.92 | 4788.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 4758.10 | 4778.70 | 4783.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4758.00 | 4737.38 | 4753.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 4774.00 | 4737.38 | 4753.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 4740.00 | 4737.90 | 4752.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 4740.60 | 4737.90 | 4752.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 4719.90 | 4699.13 | 4717.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 4719.90 | 4699.13 | 4717.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4729.80 | 4705.26 | 4718.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 4729.80 | 4705.26 | 4718.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 4713.30 | 4706.87 | 4718.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 4695.00 | 4706.87 | 4718.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 4775.60 | 4718.72 | 4721.54 | SL hit (close>static) qty=1.00 sl=4733.00 alert=retest2 |

### Cycle 230 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 4775.00 | 4729.97 | 4726.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 09:15:00 | 4840.00 | 4771.87 | 4758.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 15:15:00 | 4793.00 | 4803.31 | 4784.32 | EMA200 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 4202.05 | 2024-04-22 13:15:00 | 4143.00 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2024-04-24 14:45:00 | 4158.65 | 2024-04-25 12:15:00 | 4105.15 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-04-25 11:30:00 | 4144.00 | 2024-04-25 12:15:00 | 4105.15 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-04-26 15:15:00 | 4215.20 | 2024-05-03 13:15:00 | 4200.40 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2024-04-29 10:45:00 | 4199.95 | 2024-05-03 13:15:00 | 4200.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-04-29 15:15:00 | 4199.90 | 2024-05-03 13:15:00 | 4200.40 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-05-08 15:15:00 | 4287.75 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-05-09 09:30:00 | 4299.95 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-05-10 12:00:00 | 4285.90 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-05-10 12:30:00 | 4289.95 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-05-13 12:30:00 | 4365.00 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-05-13 14:30:00 | 4378.95 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-05-14 09:30:00 | 4369.65 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-05-14 12:30:00 | 4370.90 | 2024-05-15 11:15:00 | 4300.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-05-21 10:15:00 | 4500.05 | 2024-05-21 12:15:00 | 4336.50 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-05-23 09:30:00 | 4522.05 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2024-05-24 15:15:00 | 4530.00 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-05-27 10:15:00 | 4515.00 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-05-28 14:45:00 | 4588.80 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-05-29 13:15:00 | 4589.70 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-05-29 14:00:00 | 4596.80 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-05-30 10:30:00 | 4600.15 | 2024-05-31 10:15:00 | 4530.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-06-05 09:15:00 | 4777.00 | 2024-06-12 14:15:00 | 4900.00 | STOP_HIT | 1.00 | 2.57% |
| SELL | retest2 | 2024-06-21 11:15:00 | 4790.00 | 2024-06-27 14:15:00 | 4550.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 14:15:00 | 4764.95 | 2024-06-27 14:15:00 | 4526.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-21 11:15:00 | 4790.00 | 2024-06-28 09:15:00 | 4612.45 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2024-06-21 14:15:00 | 4764.95 | 2024-06-28 09:15:00 | 4612.45 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest2 | 2024-07-05 11:15:00 | 4622.00 | 2024-07-08 10:15:00 | 4588.05 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-07-05 12:00:00 | 4629.00 | 2024-07-08 10:15:00 | 4588.05 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-07-05 14:45:00 | 4621.45 | 2024-07-08 10:15:00 | 4588.05 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-07-15 09:15:00 | 4877.80 | 2024-07-15 11:15:00 | 4846.95 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-07-22 10:15:00 | 5078.20 | 2024-07-22 15:15:00 | 5000.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-07-29 11:30:00 | 5313.80 | 2024-08-05 14:15:00 | 5576.90 | STOP_HIT | 1.00 | 4.95% |
| BUY | retest2 | 2024-07-29 13:30:00 | 5347.00 | 2024-08-05 14:15:00 | 5576.90 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2024-08-09 09:30:00 | 5878.05 | 2024-08-13 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-12 09:30:00 | 5857.30 | 2024-08-13 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-08-12 10:00:00 | 5859.85 | 2024-08-13 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-08-12 12:00:00 | 5914.90 | 2024-08-13 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-08-13 09:15:00 | 5865.00 | 2024-08-13 09:15:00 | 5805.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-08-20 09:15:00 | 5885.95 | 2024-08-20 11:15:00 | 5840.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-08-20 10:30:00 | 5861.70 | 2024-08-20 11:15:00 | 5840.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-08-20 12:45:00 | 5890.95 | 2024-08-23 11:15:00 | 5892.35 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2024-08-27 11:15:00 | 5880.00 | 2024-08-28 11:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-08-28 10:15:00 | 5876.95 | 2024-08-28 11:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-09-11 12:45:00 | 6138.30 | 2024-09-12 10:15:00 | 6239.95 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-09-12 09:45:00 | 6150.40 | 2024-09-12 10:15:00 | 6239.95 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-10-09 09:15:00 | 5692.55 | 2024-10-16 09:15:00 | 5690.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-10-21 14:45:00 | 5593.90 | 2024-10-24 15:15:00 | 5314.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 15:15:00 | 5590.00 | 2024-10-24 15:15:00 | 5310.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 5593.90 | 2024-10-28 13:15:00 | 5315.00 | STOP_HIT | 0.50 | 4.99% |
| SELL | retest2 | 2024-10-21 15:15:00 | 5590.00 | 2024-10-28 13:15:00 | 5315.00 | STOP_HIT | 0.50 | 4.92% |
| BUY | retest2 | 2024-11-08 13:45:00 | 5295.40 | 2024-11-11 10:15:00 | 5257.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-11-08 14:45:00 | 5306.00 | 2024-11-11 10:15:00 | 5257.10 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-11-27 09:15:00 | 5296.70 | 2024-12-05 10:15:00 | 5292.60 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-11-27 11:00:00 | 5301.90 | 2024-12-05 10:15:00 | 5292.60 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-11-27 12:30:00 | 5297.80 | 2024-12-05 10:15:00 | 5292.60 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-12-02 09:30:00 | 5295.10 | 2024-12-05 10:15:00 | 5292.60 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-12-13 09:15:00 | 5056.25 | 2024-12-18 12:15:00 | 5162.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-12-24 09:15:00 | 4799.90 | 2024-12-24 13:15:00 | 4867.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-12-24 11:45:00 | 4816.00 | 2024-12-24 13:15:00 | 4867.70 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-24 15:00:00 | 4804.75 | 2024-12-26 14:15:00 | 4892.55 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest1 | 2025-01-01 11:00:00 | 5293.90 | 2025-01-03 12:15:00 | 5237.45 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest1 | 2025-01-01 13:15:00 | 5295.90 | 2025-01-03 12:15:00 | 5237.45 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest1 | 2025-01-01 14:00:00 | 5290.70 | 2025-01-03 12:15:00 | 5237.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest1 | 2025-01-01 15:15:00 | 5300.50 | 2025-01-03 12:15:00 | 5237.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-22 09:15:00 | 4906.85 | 2025-01-27 09:15:00 | 4661.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-22 09:15:00 | 4906.85 | 2025-01-28 11:15:00 | 4416.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-06 14:00:00 | 4375.25 | 2025-02-11 09:15:00 | 4156.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 09:15:00 | 4368.30 | 2025-02-11 09:15:00 | 4149.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 14:00:00 | 4375.25 | 2025-02-12 12:15:00 | 4121.90 | STOP_HIT | 0.50 | 5.79% |
| SELL | retest2 | 2025-02-07 09:15:00 | 4368.30 | 2025-02-12 12:15:00 | 4121.90 | STOP_HIT | 0.50 | 5.64% |
| SELL | retest2 | 2025-02-13 11:00:00 | 4246.25 | 2025-02-13 12:15:00 | 4258.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-02-17 09:15:00 | 4115.80 | 2025-02-19 11:15:00 | 4158.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-02-17 09:45:00 | 4099.90 | 2025-02-19 11:15:00 | 4158.90 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-02-18 09:15:00 | 4090.00 | 2025-02-19 11:15:00 | 4158.90 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-03-03 11:30:00 | 3922.20 | 2025-03-05 11:15:00 | 4034.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-03-03 12:00:00 | 3916.65 | 2025-03-05 11:15:00 | 4034.00 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-03-04 09:15:00 | 3905.05 | 2025-03-05 11:15:00 | 4034.00 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-03-28 13:15:00 | 4010.00 | 2025-04-01 13:15:00 | 4043.80 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-03-28 13:45:00 | 3999.90 | 2025-04-01 13:15:00 | 4043.80 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-03-28 15:00:00 | 3993.00 | 2025-04-01 13:15:00 | 4043.80 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-04-09 11:30:00 | 4003.10 | 2025-04-28 10:15:00 | 4216.30 | STOP_HIT | 1.00 | 5.33% |
| BUY | retest2 | 2025-04-09 12:00:00 | 4002.85 | 2025-04-28 10:15:00 | 4216.30 | STOP_HIT | 1.00 | 5.33% |
| BUY | retest1 | 2025-06-12 09:15:00 | 5870.00 | 2025-06-16 09:15:00 | 5843.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5900.50 | 2025-06-18 10:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-13 14:00:00 | 5887.50 | 2025-06-18 10:15:00 | 5815.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-06-13 14:30:00 | 5889.00 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-13 15:15:00 | 5889.00 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-17 09:15:00 | 5953.50 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-06-18 09:15:00 | 5897.50 | 2025-06-18 11:15:00 | 5803.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-06-20 11:30:00 | 5694.50 | 2025-06-26 12:15:00 | 5650.00 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2025-07-03 13:45:00 | 5777.50 | 2025-07-09 09:15:00 | 5669.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-07-18 11:30:00 | 5318.50 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-07-21 11:15:00 | 5325.50 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-07-21 13:45:00 | 5331.00 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-21 15:15:00 | 5330.00 | 2025-07-25 09:15:00 | 5407.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-01 11:30:00 | 5096.50 | 2025-08-11 15:15:00 | 5098.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-08-22 09:30:00 | 5730.00 | 2025-08-22 15:15:00 | 5641.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-09-02 11:15:00 | 5132.50 | 2025-09-03 11:15:00 | 5187.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-03 14:45:00 | 5143.00 | 2025-09-04 09:15:00 | 5214.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-10-07 10:15:00 | 5240.00 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-10-07 14:00:00 | 5236.00 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-10-07 14:45:00 | 5237.50 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-10-08 09:30:00 | 5244.50 | 2025-10-08 15:15:00 | 5161.00 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-10-09 09:15:00 | 5231.50 | 2025-10-15 14:15:00 | 5329.50 | STOP_HIT | 1.00 | 1.87% |
| SELL | retest2 | 2025-10-20 09:15:00 | 5325.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-10-20 10:00:00 | 5328.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-10-20 14:45:00 | 5322.50 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-10-21 14:30:00 | 5310.00 | 2025-10-23 10:15:00 | 5426.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-10-29 11:00:00 | 5270.00 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-29 14:00:00 | 5269.50 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-29 15:00:00 | 5264.50 | 2025-10-30 14:15:00 | 5325.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-10-31 09:45:00 | 5263.50 | 2025-11-10 13:15:00 | 5118.50 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-11-04 09:45:00 | 5163.50 | 2025-11-10 13:15:00 | 5118.50 | STOP_HIT | 1.00 | 0.87% |
| SELL | retest2 | 2025-11-19 14:15:00 | 5020.50 | 2025-11-26 10:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-11-20 15:00:00 | 5025.00 | 2025-11-26 10:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2025-11-27 09:15:00 | 5048.00 | 2025-11-27 09:15:00 | 4982.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-02 13:00:00 | 4963.00 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-03 09:15:00 | 4958.00 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-03 15:15:00 | 4965.70 | 2025-12-04 10:15:00 | 4999.60 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-18 10:30:00 | 5090.00 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-22 09:30:00 | 5087.00 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-12-22 10:30:00 | 5083.90 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-22 13:15:00 | 5083.70 | 2025-12-23 10:15:00 | 5029.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-12-29 10:30:00 | 5011.20 | 2026-01-07 09:15:00 | 4952.40 | STOP_HIT | 1.00 | 1.17% |
| SELL | retest2 | 2025-12-29 11:00:00 | 4999.40 | 2026-01-07 09:15:00 | 4952.40 | STOP_HIT | 1.00 | 0.94% |
| SELL | retest2 | 2026-01-12 09:15:00 | 4800.00 | 2026-01-21 09:15:00 | 4598.00 | PARTIAL | 0.50 | 4.21% |
| SELL | retest2 | 2026-01-12 09:15:00 | 4800.00 | 2026-01-22 09:15:00 | 4776.60 | STOP_HIT | 0.50 | 0.49% |
| SELL | retest2 | 2026-01-14 09:15:00 | 4840.00 | 2026-01-22 12:15:00 | 4727.40 | STOP_HIT | 1.00 | 2.33% |
| SELL | retest2 | 2026-02-01 12:15:00 | 4552.30 | 2026-02-03 11:15:00 | 4634.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-03-06 14:45:00 | 4848.00 | 2026-03-13 10:15:00 | 4605.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 14:45:00 | 4848.00 | 2026-03-16 09:15:00 | 4699.00 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2026-04-02 13:00:00 | 4807.60 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-04-06 10:00:00 | 4804.90 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2026-04-07 10:15:00 | 4804.40 | 2026-04-07 12:15:00 | 4773.10 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4843.60 | 2026-04-21 10:15:00 | 4886.30 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2026-05-05 15:15:00 | 4695.00 | 2026-05-06 09:15:00 | 4775.60 | STOP_HIT | 1.00 | -1.72% |
