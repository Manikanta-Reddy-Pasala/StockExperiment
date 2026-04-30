# Endurance Technologies Ltd. (ENDURANCE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2330.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 2 / 2
- **Total realized P&L (per unit):** 218.76
- **Avg P&L per closed trade:** 54.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 2438.80 | 2491.80 | 2491.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 2424.10 | 2490.60 | 2491.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 10:15:00 | 2391.10 | 2385.93 | 2426.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 09:15:00 | 2311.25 | 2389.86 | 2403.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-18 10:15:00 | 1980.55 | 1890.46 | 1976.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 2133.00 | 1960.55 | 1960.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 2187.80 | 1962.81 | 1961.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 15:15:00 | 2612.80 | 2614.79 | 2481.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-23 09:15:00 | 2655.00 | 2615.19 | 2482.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-01 13:15:00 | 2495.20 | 2598.04 | 2504.49 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 2677.40 | 2797.01 | 2797.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 2648.00 | 2794.29 | 2796.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 2696.80 | 2670.22 | 2715.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 10:15:00 | 2656.10 | 2670.34 | 2714.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2492.80 | 2494.76 | 2571.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 10:15:00 | 2481.60 | 2494.63 | 2570.95 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-11 09:15:00 | 2555.70 | 2489.06 | 2553.55 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-29 09:15:00 | 2311.25 | 2025-01-13 12:15:00 | 2034.40 | TARGET | 276.85 |
| BUY | 2025-07-23 09:15:00 | 2655.00 | 2025-08-01 13:15:00 | 2495.20 | EXIT_EMA400 | -159.80 |
| SELL | 2025-12-22 10:15:00 | 2656.10 | 2026-01-16 09:15:00 | 2480.28 | TARGET | 175.82 |
| SELL | 2026-02-03 10:15:00 | 2481.60 | 2026-02-11 09:15:00 | 2555.70 | EXIT_EMA400 | -74.10 |
