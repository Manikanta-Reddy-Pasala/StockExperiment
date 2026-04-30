# Schaeffler India Ltd. (SCHAEFFLER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4135.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 7 |
| ENTRY1 | 4 |
| ENTRY2 | 4 |
| EXIT | 4 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -659.20
- **Avg P&L per closed trade:** -82.40

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 3492.70 | 3315.39 | 3315.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 3529.00 | 3319.23 | 3317.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 3297.75 | 3343.68 | 3330.46 | EMA200 retest candle locked |

### Cycle 2 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2968.55 | 3317.67 | 3318.54 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 3594.60 | 3290.96 | 3290.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3613.00 | 3322.27 | 3306.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 15:15:00 | 4000.00 | 4010.40 | 3808.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 13:15:00 | 4051.50 | 3970.89 | 3838.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 3999.00 | 4106.26 | 3993.76 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-30 09:15:00 | 4122.50 | 4101.11 | 3995.00 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 4016.00 | 4098.97 | 4006.39 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-04 14:15:00 | 4061.90 | 4098.61 | 4006.67 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4052.00 | 4100.32 | 4014.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 4004.20 | 4099.36 | 4014.61 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3867.00 | 3970.51 | 3970.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 3858.00 | 3968.31 | 3969.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.94 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 4129.00 | 3964.19 | 3964.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 4146.40 | 3966.00 | 3964.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 14:15:00 | 4108.80 | 4016.02 | 3994.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 4049.90 | 4094.07 | 4044.85 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-13 12:15:00 | 4028.70 | 4093.42 | 4044.77 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3869.90 | 4010.42 | 4010.92 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 4294.00 | 4011.71 | 4010.91 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3860.90 | 4028.20 | 4029.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3800.80 | 3983.08 | 4003.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 3903.60 | 3901.73 | 3949.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 12:15:00 | 3824.70 | 3900.15 | 3947.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3807.70 | 3859.49 | 3907.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 14:15:00 | 3755.80 | 3856.29 | 3904.48 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | Close above EMA400 |

### Cycle 9 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 4248.30 | 3838.74 | 3838.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 4289.00 | 3847.23 | 3842.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3982.30 | 4028.11 | 3950.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 14:15:00 | 4057.00 | 3991.34 | 3942.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4000.00 | 3992.30 | 3943.70 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-20 09:15:00 | 4081.50 | 3991.73 | 3945.08 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 3954.90 | 3999.02 | 3951.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 15:15:00 | 3937.00 | 3998.40 | 3951.52 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-06-30 13:15:00 | 4051.50 | 2025-08-07 10:15:00 | 4004.20 | EXIT_EMA400 | -47.30 |
| BUY | 2025-07-30 09:15:00 | 4122.50 | 2025-08-07 10:15:00 | 4004.20 | EXIT_EMA400 | -118.30 |
| BUY | 2025-08-04 14:15:00 | 4061.90 | 2025-08-07 10:15:00 | 4004.20 | EXIT_EMA400 | -57.70 |
| BUY | 2025-09-29 14:15:00 | 4108.80 | 2025-10-13 12:15:00 | 4028.70 | EXIT_EMA400 | -80.10 |
| SELL | 2025-12-23 12:15:00 | 3824.70 | 2026-02-03 09:15:00 | 3835.90 | EXIT_EMA400 | -11.20 |
| SELL | 2026-01-09 14:15:00 | 3755.80 | 2026-02-03 09:15:00 | 3835.90 | EXIT_EMA400 | -80.10 |
| BUY | 2026-03-18 14:15:00 | 4057.00 | 2026-03-23 15:15:00 | 3937.00 | EXIT_EMA400 | -120.00 |
| BUY | 2026-03-20 09:15:00 | 4081.50 | 2026-03-23 15:15:00 | 3937.00 | EXIT_EMA400 | -144.50 |
