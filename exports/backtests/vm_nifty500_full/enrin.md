# Siemens Energy India Ltd. (ENRIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2025-06-19 09:15:00 → 2026-04-30 15:15:00 (1482 bars)
- **Last close:** 3278.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -316.00
- **Avg P&L per closed trade:** -105.33

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 3130.10 | 3260.92 | 3261.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 3125.30 | 3259.57 | 3260.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 11:15:00 | 3236.20 | 3232.80 | 3245.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 3170.00 | 3232.68 | 3245.11 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 3170.00 | 3232.68 | 3245.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-06 10:15:00 | 3160.00 | 3231.95 | 3244.69 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3226.00 | 3227.97 | 3241.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-10 09:15:00 | 3268.80 | 3228.33 | 3241.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 3349.00 | 3252.85 | 3252.49 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 10:15:00 | 3121.00 | 3253.56 | 3253.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-26 11:15:00 | 3110.70 | 3252.14 | 3253.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 2524.90 | 2485.82 | 2686.82 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 15:15:00 | 2931.50 | 2748.27 | 2747.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 2953.20 | 2774.02 | 2761.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 2792.00 | 2801.22 | 2777.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 09:15:00 | 2882.10 | 2797.14 | 2777.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2882.10 | 2797.14 | 2777.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 12:15:00 | 2773.70 | 2799.56 | 2779.62 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 2576.90 | 2762.92 | 2763.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2560.00 | 2760.90 | 2762.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 2770.30 | 2722.28 | 2740.91 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 13:15:00 | 2909.30 | 2756.82 | 2756.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 2963.50 | 2761.81 | 2759.09 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-11-06 09:15:00 | 3170.00 | 2025-11-10 09:15:00 | 3268.80 | EXIT_EMA400 | -98.80 |
| SELL | 2025-11-06 10:15:00 | 3160.00 | 2025-11-10 09:15:00 | 3268.80 | EXIT_EMA400 | -108.80 |
| BUY | 2026-03-18 09:15:00 | 2882.10 | 2026-03-19 12:15:00 | 2773.70 | EXIT_EMA400 | -108.40 |
