# Mahindra & Mahindra Ltd. (M&M.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 3097.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 3 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / EMA400 exits:** 5 / 2
- **Total realized P&L (per unit):** 860.11
- **Avg P&L per closed trade:** 122.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-03 10:15:00 | 1476.10 | 1534.06 | 1534.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-03 11:15:00 | 1469.90 | 1533.42 | 1533.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-09 09:15:00 | 1527.55 | 1521.86 | 1527.62 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 10:15:00 | 1545.85 | 1531.86 | 1531.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 11:15:00 | 1565.75 | 1533.61 | 1532.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-21 11:15:00 | 1631.80 | 1632.77 | 1596.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-26 09:15:00 | 1664.65 | 1633.43 | 1598.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1620.90 | 1652.31 | 1619.93 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-08 13:15:00 | 1618.70 | 1651.98 | 1619.93 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 10:15:00 | 2782.70 | 2975.60 | 2975.69 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 14:15:00 | 3174.00 | 2973.74 | 2973.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 3189.00 | 2977.81 | 2975.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2978.55 | 3037.60 | 3009.41 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-19 12:15:00 | 2756.00 | 2987.75 | 2987.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 09:15:00 | 2695.00 | 2967.89 | 2977.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 2788.40 | 2782.02 | 2855.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-24 09:15:00 | 2739.20 | 2789.78 | 2851.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-21 12:15:00 | 2759.00 | 2683.61 | 2758.40 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 14:15:00 | 3071.40 | 2808.08 | 2806.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 3097.30 | 2816.09 | 2811.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 2977.50 | 2983.44 | 2922.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 10:15:00 | 3018.40 | 2983.46 | 2925.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2989.30 | 3020.92 | 2962.15 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 13:15:00 | 3003.30 | 3019.86 | 2962.78 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3095.20 | 3119.00 | 3056.90 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 09:15:00 | 3119.10 | 3117.37 | 3058.21 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3222.70 | 3270.71 | 3199.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-29 14:15:00 | 3193.10 | 3267.72 | 3199.91 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3380.00 | 3618.44 | 3619.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 10:15:00 | 3349.00 | 3615.76 | 3618.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 09:15:00 | 3592.10 | 3575.65 | 3596.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-16 09:15:00 | 3529.40 | 3590.36 | 3599.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 3284.50 | 3175.60 | 3287.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 09:15:00 | 3242.50 | 3180.74 | 3286.56 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-26 09:15:00 | 1664.65 | 2024-01-08 13:15:00 | 1618.70 | EXIT_EMA400 | -45.95 |
| SELL | 2025-03-24 09:15:00 | 2739.20 | 2025-04-21 12:15:00 | 2759.00 | EXIT_EMA400 | -19.80 |
| BUY | 2025-06-13 13:15:00 | 3003.30 | 2025-06-20 09:15:00 | 3124.86 | TARGET | 121.56 |
| BUY | 2025-06-02 10:15:00 | 3018.40 | 2025-07-23 09:15:00 | 3298.29 | TARGET | 279.89 |
| BUY | 2025-07-15 09:15:00 | 3119.10 | 2025-07-23 09:15:00 | 3301.78 | TARGET | 182.68 |
| SELL | 2026-02-16 09:15:00 | 3529.40 | 2026-03-02 09:15:00 | 3319.84 | TARGET | 209.56 |
| SELL | 2026-04-16 09:15:00 | 3242.50 | 2026-04-23 09:15:00 | 3110.33 | TARGET | 132.17 |
