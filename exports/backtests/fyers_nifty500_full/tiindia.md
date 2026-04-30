# Tube Investments of India Ltd. (TIINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2963.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 970.81
- **Avg P&L per closed trade:** 107.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 09:15:00 | 3925.95 | 4055.06 | 4055.18 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 4229.35 | 4051.48 | 4051.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 14:15:00 | 4287.75 | 4057.86 | 4054.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 12:15:00 | 4089.05 | 4121.16 | 4091.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 09:15:00 | 4208.50 | 4107.43 | 4086.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 4208.50 | 4107.43 | 4086.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-09 10:15:00 | 4297.15 | 4109.32 | 4087.67 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 4176.95 | 4119.30 | 4094.22 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-11 10:15:00 | 4196.35 | 4120.07 | 4094.73 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-11-05 09:15:00 | 4181.35 | 4360.51 | 4255.85 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 3483.90 | 4175.08 | 4176.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 3449.20 | 4154.34 | 4165.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 3751.00 | 3746.79 | 3877.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 12:15:00 | 3705.05 | 3746.23 | 3872.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2799.00 | 2669.03 | 2806.54 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-30 09:15:00 | 2857.90 | 2676.63 | 2806.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 3001.00 | 2876.90 | 2876.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 3033.20 | 2879.68 | 2877.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 2978.00 | 2992.28 | 2949.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 10:15:00 | 3001.80 | 2988.01 | 2949.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3001.80 | 2988.01 | 2949.53 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-16 13:15:00 | 3004.60 | 2988.37 | 2950.28 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-17 09:15:00 | 2950.20 | 2987.90 | 2950.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 11:15:00 | 2852.10 | 2941.23 | 2941.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 2836.60 | 2940.19 | 2941.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 2917.50 | 2913.41 | 2926.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-05 09:15:00 | 2883.00 | 2912.59 | 2925.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 2878.70 | 2911.70 | 2924.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-06 11:15:00 | 2941.90 | 2911.57 | 2924.54 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 09:15:00 | 3056.30 | 2935.40 | 2934.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 3076.00 | 2944.18 | 2939.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 2991.90 | 3004.27 | 2974.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-01 11:15:00 | 3072.10 | 3000.35 | 2974.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 2986.00 | 3012.43 | 2984.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-05 11:15:00 | 2970.20 | 3012.01 | 2984.55 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 15:15:00 | 2970.60 | 3099.39 | 3100.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 2966.00 | 3096.80 | 3098.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3083.30 | 3082.88 | 3090.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-18 09:15:00 | 3030.50 | 3083.13 | 3090.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 14:15:00 | 2639.80 | 2435.99 | 2573.96 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 10:15:00 | 2738.80 | 2571.21 | 2571.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 14:15:00 | 2742.20 | 2577.78 | 2574.34 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-11 10:15:00 | 4196.35 | 2024-10-16 14:15:00 | 4501.22 | TARGET | 304.87 |
| BUY | 2024-10-09 09:15:00 | 4208.50 | 2024-10-23 14:15:00 | 4574.15 | TARGET | 365.65 |
| BUY | 2024-10-09 10:15:00 | 4297.15 | 2024-11-05 09:15:00 | 4181.35 | EXIT_EMA400 | -115.80 |
| SELL | 2024-12-17 12:15:00 | 3705.05 | 2025-01-27 09:15:00 | 3202.94 | TARGET | 502.11 |
| BUY | 2025-06-16 10:15:00 | 3001.80 | 2025-06-17 09:15:00 | 2950.20 | EXIT_EMA400 | -51.60 |
| BUY | 2025-06-16 13:15:00 | 3004.60 | 2025-06-17 09:15:00 | 2950.20 | EXIT_EMA400 | -54.40 |
| SELL | 2025-08-05 09:15:00 | 2883.00 | 2025-08-06 11:15:00 | 2941.90 | EXIT_EMA400 | -58.90 |
| BUY | 2025-09-01 11:15:00 | 3072.10 | 2025-09-05 11:15:00 | 2970.20 | EXIT_EMA400 | -101.90 |
| SELL | 2025-11-18 09:15:00 | 3030.50 | 2025-11-27 11:15:00 | 2849.71 | TARGET | 180.79 |
