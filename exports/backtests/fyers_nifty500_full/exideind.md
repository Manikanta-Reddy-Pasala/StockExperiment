# Exide Industries Ltd. (EXIDEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 359.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 3 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -2.54
- **Avg P&L per closed trade:** -0.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 497.30 | 517.82 | 517.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 495.10 | 517.39 | 517.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 10:15:00 | 487.60 | 484.64 | 495.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 12:15:00 | 478.90 | 489.46 | 496.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 490.20 | 489.22 | 495.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-09 09:15:00 | 500.65 | 489.49 | 495.80 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 394.50 | 372.82 | 372.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 395.85 | 382.56 | 378.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 386.55 | 386.87 | 381.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-27 09:15:00 | 392.50 | 384.98 | 381.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 384.20 | 385.50 | 382.57 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-02 11:15:00 | 385.15 | 385.50 | 382.58 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 383.00 | 385.46 | 382.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-03 10:15:00 | 381.90 | 385.35 | 382.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 376.00 | 383.51 | 383.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 373.85 | 383.41 | 383.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 383.50 | 382.33 | 382.92 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 396.55 | 383.60 | 383.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 11:15:00 | 397.35 | 383.74 | 383.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 12:15:00 | 408.25 | 408.50 | 399.73 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 379.60 | 397.05 | 397.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 378.20 | 392.44 | 394.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 14:15:00 | 380.25 | 379.94 | 385.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-03 09:15:00 | 374.40 | 379.81 | 385.42 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 10:15:00 | 322.75 | 309.74 | 321.89 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 13:15:00 | 365.35 | 328.88 | 328.72 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-07 12:15:00 | 478.90 | 2024-10-09 09:15:00 | 500.65 | EXIT_EMA400 | -21.75 |
| BUY | 2025-06-27 09:15:00 | 392.50 | 2025-07-03 10:15:00 | 381.90 | EXIT_EMA400 | -10.60 |
| BUY | 2025-07-02 11:15:00 | 385.15 | 2025-07-03 10:15:00 | 381.90 | EXIT_EMA400 | -3.25 |
| SELL | 2025-12-03 09:15:00 | 374.40 | 2026-01-12 09:15:00 | 341.34 | TARGET | 33.06 |
