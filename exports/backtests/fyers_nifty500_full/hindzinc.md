# Hindustan Zinc Ltd. (HINDZINC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 596.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -49.90
- **Avg P&L per closed trade:** -16.63

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 567.20 | 630.14 | 630.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 09:15:00 | 537.90 | 626.23 | 628.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 516.45 | 514.52 | 545.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 09:15:00 | 507.50 | 516.66 | 540.82 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 529.60 | 512.58 | 529.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-10-24 09:15:00 | 534.70 | 513.53 | 529.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 09:15:00 | 451.20 | 438.84 | 438.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 455.35 | 439.00 | 438.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 459.90 | 476.96 | 461.61 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 422.55 | 453.45 | 453.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 09:15:00 | 419.70 | 442.02 | 446.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 431.70 | 431.35 | 438.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 15:15:00 | 430.80 | 431.38 | 437.82 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 461.20 | 439.50 | 439.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 465.95 | 442.46 | 440.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 476.55 | 480.39 | 467.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 09:15:00 | 483.50 | 479.88 | 467.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 14:15:00 | 547.35 | 591.31 | 591.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 09:15:00 | 531.10 | 590.27 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 557.60 | 548.24 | 565.06 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 11:15:00 | 621.05 | 573.70 | 573.60 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-07 09:15:00 | 507.50 | 2024-10-24 09:15:00 | 534.70 | EXIT_EMA400 | -27.20 |
| SELL | 2025-08-21 15:15:00 | 430.80 | 2025-09-01 09:15:00 | 438.00 | EXIT_EMA400 | -7.20 |
| BUY | 2025-10-29 09:15:00 | 483.50 | 2025-11-06 09:15:00 | 468.00 | EXIT_EMA400 | -15.50 |
