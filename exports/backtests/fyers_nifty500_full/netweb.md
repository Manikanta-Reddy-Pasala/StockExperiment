# Netweb Technologies India Ltd. (NETWEB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 4069.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 0 / 4
- **Target hits / EMA400 exits:** 0 / 4
- **Total realized P&L (per unit):** -332.40
- **Avg P&L per closed trade:** -83.10

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 2442.00 | 2725.72 | 2726.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 2373.70 | 2704.38 | 2715.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1609.80 | 1588.89 | 1811.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 15:15:00 | 1545.90 | 1592.60 | 1789.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-05 09:15:00 | 1675.90 | 1505.99 | 1626.30 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1894.20 | 1679.48 | 1679.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1935.90 | 1684.28 | 1681.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1841.50 | 1843.83 | 1779.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 09:15:00 | 1929.80 | 1821.56 | 1796.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 1855.10 | 1883.34 | 1842.10 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-28 13:15:00 | 1826.80 | 1882.77 | 1842.02 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 14:15:00 | 3259.90 | 3283.65 | 3283.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 3206.80 | 3277.53 | 3280.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 3252.80 | 3211.58 | 3243.46 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 3494.10 | 3269.87 | 3268.86 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 09:15:00 | 3133.00 | 3267.58 | 3268.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 3109.40 | 3253.51 | 3260.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 3217.10 | 3214.38 | 3238.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 12:15:00 | 3152.50 | 3219.01 | 3239.75 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3229.40 | 3218.11 | 3238.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 10:15:00 | 3247.90 | 3218.41 | 3238.93 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 3660.30 | 3237.74 | 3237.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 3681.90 | 3285.75 | 3262.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 09:15:00 | 3373.20 | 3402.70 | 3330.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 3176.60 | 3289.30 | 3289.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3135.80 | 3282.16 | 3285.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3341.30 | 3252.59 | 3269.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 14:15:00 | 3311.00 | 3257.03 | 3270.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 3311.00 | 3257.03 | 3270.85 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 15:15:00 | 3315.00 | 3257.60 | 3271.07 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 3562.90 | 3284.36 | 3283.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 3694.00 | 3296.52 | 3289.70 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-26 15:15:00 | 1545.90 | 2025-05-05 09:15:00 | 1675.90 | EXIT_EMA400 | -130.00 |
| BUY | 2025-07-11 09:15:00 | 1929.80 | 2025-07-28 13:15:00 | 1826.80 | EXIT_EMA400 | -103.00 |
| SELL | 2026-02-02 12:15:00 | 3152.50 | 2026-02-03 10:15:00 | 3247.90 | EXIT_EMA400 | -95.40 |
| SELL | 2026-04-08 14:15:00 | 3311.00 | 2026-04-08 15:15:00 | 3315.00 | EXIT_EMA400 | -4.00 |
