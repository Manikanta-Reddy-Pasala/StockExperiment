# Hindustan Zinc Ltd. (HINDZINC.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 595.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -35.90
- **Avg P&L per closed trade:** -5.98

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 297.00 | 315.91 | 316.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 292.20 | 312.99 | 314.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 09:15:00 | 304.30 | 303.48 | 307.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-22 11:15:00 | 300.35 | 303.76 | 307.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-12-04 09:15:00 | 314.30 | 302.67 | 305.88 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 324.55 | 308.68 | 308.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 15:15:00 | 326.60 | 309.61 | 309.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-19 09:15:00 | 312.10 | 312.62 | 310.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-28 10:15:00 | 315.40 | 311.90 | 310.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-10 09:15:00 | 312.40 | 314.18 | 312.41 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 13:15:00 | 310.45 | 313.21 | 313.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 308.80 | 313.03 | 313.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 09:15:00 | 312.30 | 312.03 | 312.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-04 10:15:00 | 311.25 | 312.02 | 312.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-03-04 13:15:00 | 312.75 | 312.03 | 312.56 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 10:15:00 | 345.20 | 310.16 | 310.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 12:15:00 | 348.20 | 310.89 | 310.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 14:15:00 | 657.95 | 658.46 | 606.47 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 511.50 | 602.08 | 602.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 09:15:00 | 497.15 | 574.08 | 586.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 516.60 | 514.37 | 539.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-07 09:15:00 | 507.50 | 516.55 | 536.36 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-23 09:15:00 | 529.60 | 512.55 | 527.06 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 10:15:00 | 455.35 | 439.01 | 438.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 461.65 | 439.92 | 439.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 459.90 | 476.95 | 461.63 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 422.55 | 453.45 | 453.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 11:15:00 | 421.45 | 453.13 | 453.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 12:15:00 | 446.20 | 446.04 | 449.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 15:15:00 | 442.25 | 445.87 | 449.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-01 09:15:00 | 438.00 | 430.31 | 436.20 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 461.20 | 439.49 | 439.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 09:15:00 | 465.95 | 442.46 | 440.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 476.55 | 480.40 | 467.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 09:15:00 | 483.65 | 479.89 | 467.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-06 09:15:00 | 468.00 | 479.26 | 468.95 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 15:15:00 | 547.30 | 591.38 | 591.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-18 09:15:00 | 531.10 | 590.78 | 591.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 557.60 | 548.46 | 565.21 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 616.80 | 573.91 | 573.81 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-22 11:15:00 | 300.35 | 2023-12-04 09:15:00 | 314.30 | EXIT_EMA400 | -13.95 |
| BUY | 2023-12-28 10:15:00 | 315.40 | 2024-01-10 09:15:00 | 312.40 | EXIT_EMA400 | -3.00 |
| SELL | 2024-03-04 10:15:00 | 311.25 | 2024-03-04 13:15:00 | 312.75 | EXIT_EMA400 | -1.50 |
| SELL | 2024-10-07 09:15:00 | 507.50 | 2024-10-23 09:15:00 | 529.60 | EXIT_EMA400 | -22.10 |
| SELL | 2025-07-22 15:15:00 | 442.25 | 2025-07-31 15:15:00 | 421.95 | TARGET | 20.30 |
| BUY | 2025-10-29 09:15:00 | 483.65 | 2025-11-06 09:15:00 | 468.00 | EXIT_EMA400 | -15.65 |
