# Apollo Tyres Ltd. (APOLLOTYRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 409.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 3 / 2
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 20.02
- **Avg P&L per closed trade:** 4.00

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 10:15:00 | 495.40 | 508.07 | 508.12 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 519.00 | 508.06 | 508.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 520.00 | 508.27 | 508.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 512.05 | 512.35 | 510.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 14:15:00 | 518.65 | 512.29 | 510.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-04 09:15:00 | 516.50 | 524.15 | 517.69 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 12:15:00 | 501.45 | 513.84 | 513.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 495.30 | 513.66 | 513.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 12:15:00 | 492.60 | 491.53 | 499.47 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 09:15:00 | 540.05 | 504.56 | 504.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 09:15:00 | 548.60 | 507.16 | 505.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 523.65 | 527.07 | 519.25 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 15:15:00 | 466.00 | 513.72 | 513.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 452.65 | 513.11 | 513.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 413.25 | 411.06 | 432.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 387.55 | 418.36 | 429.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-11 14:15:00 | 427.75 | 415.22 | 426.24 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 13:15:00 | 457.85 | 434.08 | 434.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 460.60 | 434.59 | 434.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 475.55 | 475.82 | 462.19 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 445.95 | 457.99 | 458.03 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 10:15:00 | 459.70 | 457.97 | 457.97 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 11:15:00 | 457.85 | 457.97 | 457.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 12:15:00 | 456.75 | 457.96 | 457.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 13:15:00 | 459.10 | 457.97 | 457.97 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 459.50 | 457.98 | 457.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 12:15:00 | 465.40 | 458.14 | 458.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 459.90 | 460.38 | 459.24 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 450.20 | 458.42 | 458.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 448.20 | 457.25 | 457.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 456.10 | 448.07 | 452.21 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 467.75 | 455.33 | 455.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 485.00 | 455.75 | 455.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 475.30 | 476.68 | 469.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-26 13:15:00 | 480.50 | 476.72 | 469.22 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 467.50 | 476.60 | 469.35 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 491.15 | 505.55 | 505.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 487.10 | 504.58 | 505.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 503.00 | 502.82 | 504.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-13 09:15:00 | 496.00 | 504.33 | 504.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 446.30 | 431.91 | 448.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-20 15:15:00 | 440.40 | 433.45 | 447.88 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-09-19 14:15:00 | 518.65 | 2024-09-25 09:15:00 | 543.22 | TARGET | 24.57 |
| SELL | 2025-04-07 09:15:00 | 387.55 | 2025-04-11 14:15:00 | 427.75 | EXIT_EMA400 | -40.20 |
| BUY | 2025-09-26 13:15:00 | 480.50 | 2025-09-29 11:15:00 | 467.50 | EXIT_EMA400 | -13.00 |
| SELL | 2026-02-13 09:15:00 | 496.00 | 2026-02-19 12:15:00 | 469.79 | TARGET | 26.21 |
| SELL | 2026-04-20 15:15:00 | 440.40 | 2026-04-30 09:15:00 | 417.96 | TARGET | 22.44 |
