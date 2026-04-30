# Sarda Energy and Minerals Ltd. (SARDAEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 591.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -70.31
- **Avg P&L per closed trade:** -10.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 425.30 | 451.00 | 451.00 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 472.70 | 450.93 | 450.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 474.75 | 451.56 | 451.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 450.80 | 452.79 | 451.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-04 09:15:00 | 462.50 | 452.65 | 451.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 462.50 | 452.65 | 451.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-02-04 14:15:00 | 451.75 | 452.72 | 451.90 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 437.60 | 480.57 | 480.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 433.00 | 477.89 | 479.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 467.00 | 462.86 | 470.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-26 09:15:00 | 439.45 | 464.95 | 470.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 454.10 | 451.29 | 459.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-12 13:15:00 | 447.25 | 451.52 | 459.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 457.60 | 451.17 | 458.22 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-19 11:15:00 | 449.05 | 451.14 | 458.18 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-11 09:15:00 | 460.40 | 440.27 | 448.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 541.20 | 451.02 | 450.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 549.90 | 455.48 | 453.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 568.80 | 571.98 | 539.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-19 09:15:00 | 575.50 | 571.97 | 539.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 548.50 | 573.69 | 547.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-30 11:15:00 | 547.30 | 573.43 | 547.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 518.75 | 544.58 | 544.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 515.95 | 544.30 | 544.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 507.85 | 502.32 | 516.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 493.10 | 511.73 | 516.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 499.25 | 489.40 | 500.00 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 500.55 | 489.62 | 500.00 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 15:15:00 | 534.00 | 505.57 | 505.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 536.95 | 506.99 | 506.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 519.15 | 524.56 | 516.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 10:15:00 | 533.55 | 523.23 | 516.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 518.00 | 523.44 | 516.96 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 10:15:00 | 516.35 | 523.37 | 516.96 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2025-02-04 09:15:00 | 462.50 | 2025-02-04 14:15:00 | 451.75 | EXIT_EMA400 | -10.75 |
| SELL | 2025-06-19 11:15:00 | 449.05 | 2025-06-23 10:15:00 | 421.66 | TARGET | 27.39 |
| SELL | 2025-05-26 09:15:00 | 439.45 | 2025-07-11 09:15:00 | 460.40 | EXIT_EMA400 | -20.95 |
| SELL | 2025-06-12 13:15:00 | 447.25 | 2025-07-11 09:15:00 | 460.40 | EXIT_EMA400 | -13.15 |
| BUY | 2025-09-19 09:15:00 | 575.50 | 2025-09-30 11:15:00 | 547.30 | EXIT_EMA400 | -28.20 |
| SELL | 2026-01-08 10:15:00 | 493.10 | 2026-02-03 11:15:00 | 500.55 | EXIT_EMA400 | -7.45 |
| BUY | 2026-03-18 10:15:00 | 533.55 | 2026-03-19 10:15:00 | 516.35 | EXIT_EMA400 | -17.20 |
