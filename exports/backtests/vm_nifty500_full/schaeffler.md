# Schaeffler India Ltd. (SCHAEFFLER.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 4124.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT3 | 10 |
| ENTRY1 | 8 |
| ENTRY2 | 5 |
| EXIT | 7 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / EMA400 exits:** 4 / 9
- **Total realized P&L (per unit):** -82.16
- **Avg P&L per closed trade:** -6.32

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 2989.00 | 3146.41 | 3146.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 2961.00 | 3144.57 | 3146.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-01 09:15:00 | 2859.85 | 2854.05 | 2941.95 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 3156.75 | 2984.81 | 2984.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 13:15:00 | 3159.95 | 2986.55 | 2985.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 12:15:00 | 3182.60 | 3185.11 | 3114.24 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-20 11:15:00 | 2962.30 | 3087.53 | 3087.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 12:15:00 | 2947.00 | 3086.14 | 3087.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 10:15:00 | 2930.00 | 2927.11 | 2979.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-26 14:15:00 | 2878.50 | 2928.93 | 2976.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 2961.15 | 2921.01 | 2968.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 14:15:00 | 2973.35 | 2922.79 | 2968.40 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 3254.35 | 3006.04 | 3006.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 09:15:00 | 3296.55 | 3044.32 | 3026.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 12:15:00 | 3928.25 | 4065.93 | 3732.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-04 13:15:00 | 4293.05 | 4068.19 | 3734.91 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-10 10:15:00 | 4199.45 | 4481.87 | 4212.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 14:15:00 | 4021.15 | 4105.94 | 4106.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 3930.05 | 4103.23 | 4104.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 13:15:00 | 4073.85 | 4061.39 | 4081.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 09:15:00 | 4044.05 | 4064.89 | 4081.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 4044.05 | 4064.89 | 4081.46 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-08-26 10:15:00 | 4025.00 | 4064.49 | 4081.18 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 3893.90 | 3991.00 | 4034.82 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-06 10:15:00 | 3885.00 | 3989.95 | 4034.07 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 3928.85 | 3946.72 | 4001.83 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-16 10:15:00 | 3881.50 | 3946.07 | 4001.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 3960.90 | 3915.88 | 3974.40 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-24 10:15:00 | 3978.50 | 3916.50 | 3974.42 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 3492.70 | 3315.44 | 3315.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 09:15:00 | 3530.00 | 3319.33 | 3317.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 3297.75 | 3343.74 | 3330.61 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 2968.55 | 3317.78 | 3318.71 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 3594.60 | 3291.04 | 3290.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3610.10 | 3322.30 | 3306.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 3995.60 | 4010.40 | 3809.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 13:15:00 | 4051.50 | 3971.64 | 3839.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 4000.00 | 4105.44 | 3994.04 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-29 14:15:00 | 4034.80 | 4101.66 | 3994.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4052.00 | 4100.31 | 4014.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-07 10:15:00 | 4004.20 | 4099.35 | 4014.72 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3867.00 | 3970.42 | 3970.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 3858.00 | 3968.22 | 3969.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3951.60 | 3935.92 | 3951.99 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 4129.00 | 3964.41 | 3964.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 4146.40 | 3966.23 | 3965.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 4012.80 | 4017.43 | 3994.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 14:15:00 | 4108.80 | 4016.39 | 3994.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 4049.90 | 4094.73 | 4045.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-13 12:15:00 | 4028.70 | 4094.07 | 4045.26 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3869.90 | 4010.78 | 4011.30 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 4294.00 | 4012.01 | 4011.26 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3860.90 | 4028.45 | 4029.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3800.80 | 3983.61 | 4004.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 3903.60 | 3901.76 | 3949.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 12:15:00 | 3824.70 | 3900.19 | 3947.60 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3807.70 | 3859.52 | 3907.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 14:15:00 | 3755.80 | 3856.32 | 3904.58 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 09:15:00 | 3835.90 | 3733.97 | 3812.06 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 4245.90 | 3846.21 | 3845.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 4289.90 | 3850.63 | 3847.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3982.30 | 4029.67 | 3954.14 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 14:15:00 | 4056.70 | 3992.46 | 3945.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 3954.90 | 3999.85 | 3953.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-23 15:15:00 | 3935.00 | 3999.20 | 3953.90 | Close below EMA400 |

### Cycle 15 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 3895.20 | 3926.98 | 3927.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 3876.00 | 3926.47 | 3926.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 3968.80 | 3926.41 | 3926.77 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 3961.10 | 3927.42 | 3927.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 3979.60 | 3928.27 | 3927.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 12:15:00 | 3918.70 | 3929.02 | 3928.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-17 09:15:00 | 4007.30 | 3930.68 | 3928.95 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-26 14:15:00 | 2878.50 | 2024-04-01 14:15:00 | 2973.35 | EXIT_EMA400 | -94.85 |
| BUY | 2024-06-04 13:15:00 | 4293.05 | 2024-07-10 10:15:00 | 4199.45 | EXIT_EMA400 | -93.60 |
| SELL | 2024-08-26 09:15:00 | 4044.05 | 2024-08-27 13:15:00 | 3931.82 | TARGET | 112.23 |
| SELL | 2024-08-26 10:15:00 | 4025.00 | 2024-09-02 12:15:00 | 3856.47 | TARGET | 168.53 |
| SELL | 2024-09-06 10:15:00 | 3885.00 | 2024-09-24 10:15:00 | 3978.50 | EXIT_EMA400 | -93.50 |
| SELL | 2024-09-16 10:15:00 | 3881.50 | 2024-09-24 10:15:00 | 3978.50 | EXIT_EMA400 | -97.00 |
| BUY | 2025-07-29 14:15:00 | 4034.80 | 2025-08-05 11:15:00 | 4156.17 | TARGET | 121.37 |
| BUY | 2025-06-30 13:15:00 | 4051.50 | 2025-08-07 10:15:00 | 4004.20 | EXIT_EMA400 | -47.30 |
| BUY | 2025-09-29 14:15:00 | 4108.80 | 2025-10-13 12:15:00 | 4028.70 | EXIT_EMA400 | -80.10 |
| SELL | 2025-12-23 12:15:00 | 3824.70 | 2026-02-03 09:15:00 | 3835.90 | EXIT_EMA400 | -11.20 |
| SELL | 2026-01-09 14:15:00 | 3755.80 | 2026-02-03 09:15:00 | 3835.90 | EXIT_EMA400 | -80.10 |
| BUY | 2026-03-18 14:15:00 | 4056.70 | 2026-03-23 15:15:00 | 3935.00 | EXIT_EMA400 | -121.70 |
| BUY | 2026-04-17 09:15:00 | 4007.30 | 2026-04-22 09:15:00 | 4242.35 | TARGET | 235.05 |
