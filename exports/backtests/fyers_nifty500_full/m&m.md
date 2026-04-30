# Mahindra & Mahindra Ltd. (M&M.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3094.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT3 | 5 |
| ENTRY1 | 3 |
| ENTRY2 | 3 |
| EXIT | 2 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 5 / 1
- **Target hits / EMA400 exits:** 5 / 1
- **Total realized P&L (per unit):** 938.57
- **Avg P&L per closed trade:** 156.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 2782.70 | 2975.66 | 2976.02 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 12:15:00 | 3170.05 | 2975.77 | 2974.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 13:15:00 | 3181.35 | 2977.82 | 2975.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.60 | 3041.41 | 3011.95 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 13:15:00 | 2759.80 | 2988.04 | 2988.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2694.05 | 2970.13 | 2979.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.65 | 2856.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-24 09:15:00 | 2739.20 | 2790.34 | 2852.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 2759.00 | 2684.01 | 2759.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 13:15:00 | 2766.50 | 2684.83 | 2759.11 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.28 | 2807.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.19 | 2811.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.48 | 2922.95 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 10:15:00 | 3018.70 | 2983.47 | 2925.31 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.83 | 2962.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 13:15:00 | 3003.40 | 3019.76 | 2962.87 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3095.20 | 3119.02 | 3056.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 09:15:00 | 3119.10 | 3117.40 | 3058.29 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3223.10 | 3270.93 | 3199.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-29 14:15:00 | 3193.10 | 3267.93 | 3200.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.61 | 3619.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 11:15:00 | 3333.20 | 3613.12 | 3616.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3566.84 | 3590.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-16 10:15:00 | 3510.60 | 3584.60 | 3594.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 3284.30 | 3175.22 | 3286.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 09:15:00 | 3242.50 | 3180.38 | 3285.49 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-24 09:15:00 | 2739.20 | 2025-04-21 13:15:00 | 2766.50 | EXIT_EMA400 | -27.30 |
| BUY | 2025-06-13 13:15:00 | 3003.40 | 2025-06-20 09:15:00 | 3124.98 | TARGET | 121.58 |
| BUY | 2025-06-02 10:15:00 | 3018.70 | 2025-07-23 09:15:00 | 3298.87 | TARGET | 280.17 |
| BUY | 2025-07-15 09:15:00 | 3119.10 | 2025-07-23 09:15:00 | 3301.53 | TARGET | 182.43 |
| SELL | 2026-02-16 10:15:00 | 3510.60 | 2026-03-04 09:15:00 | 3257.88 | TARGET | 252.72 |
| SELL | 2026-04-16 09:15:00 | 3242.50 | 2026-04-23 09:15:00 | 3113.53 | TARGET | 128.97 |
