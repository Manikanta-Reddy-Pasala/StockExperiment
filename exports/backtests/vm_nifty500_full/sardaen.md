# Sarda Energy and Minerals Ltd. (SARDAEN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 591.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -107.46
- **Avg P&L per closed trade:** -11.94

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 11:15:00 | 220.30 | 237.46 | 237.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 14:15:00 | 217.45 | 233.42 | 235.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 09:15:00 | 220.00 | 209.11 | 218.72 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 14:15:00 | 232.60 | 220.78 | 220.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 10:15:00 | 235.75 | 221.15 | 220.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 241.20 | 252.65 | 241.28 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 227.86 | 235.09 | 235.11 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 15:15:00 | 248.00 | 235.05 | 235.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 09:15:00 | 253.80 | 235.24 | 235.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 12:15:00 | 263.00 | 263.21 | 253.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 09:15:00 | 264.55 | 263.12 | 253.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 10:15:00 | 252.50 | 263.02 | 253.60 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 13:15:00 | 430.50 | 450.56 | 450.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 14:15:00 | 424.65 | 450.31 | 450.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 15:15:00 | 449.75 | 449.58 | 450.14 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 472.70 | 450.84 | 450.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 474.75 | 451.48 | 451.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 11:15:00 | 450.80 | 451.77 | 451.24 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-02-04 09:15:00 | 462.50 | 451.68 | 451.20 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 462.50 | 451.68 | 451.20 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-06 09:15:00 | 471.65 | 452.61 | 451.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 459.70 | 453.32 | 452.11 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-07 12:15:00 | 460.85 | 453.45 | 452.19 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-02-11 09:15:00 | 430.20 | 454.34 | 452.72 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 437.60 | 480.53 | 480.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 433.00 | 477.85 | 479.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-19 09:15:00 | 467.70 | 462.81 | 470.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-26 09:15:00 | 439.45 | 464.86 | 470.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 457.60 | 451.10 | 458.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-19 11:15:00 | 449.05 | 451.08 | 458.10 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-11 09:15:00 | 460.70 | 440.26 | 448.49 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 541.05 | 451.03 | 450.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 549.90 | 455.49 | 453.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 14:15:00 | 568.80 | 571.97 | 539.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-19 09:15:00 | 575.75 | 571.98 | 539.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 548.50 | 573.69 | 547.70 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-30 11:15:00 | 547.30 | 573.43 | 547.70 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 12:15:00 | 518.75 | 544.53 | 544.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 515.95 | 544.25 | 544.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 507.85 | 502.31 | 516.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 493.10 | 511.71 | 516.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 499.25 | 490.05 | 500.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 12:15:00 | 501.00 | 490.36 | 500.68 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 525.45 | 506.23 | 506.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 537.00 | 507.23 | 506.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 519.15 | 524.68 | 517.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 10:15:00 | 533.55 | 523.27 | 516.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 518.00 | 523.48 | 517.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-19 10:15:00 | 516.35 | 523.41 | 517.18 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-08-05 09:15:00 | 264.55 | 2024-08-05 10:15:00 | 252.50 | EXIT_EMA400 | -12.05 |
| BUY | 2025-02-07 12:15:00 | 460.85 | 2025-02-10 09:15:00 | 486.83 | TARGET | 25.98 |
| BUY | 2025-02-04 09:15:00 | 462.50 | 2025-02-11 09:15:00 | 430.20 | EXIT_EMA400 | -32.30 |
| BUY | 2025-02-06 09:15:00 | 471.65 | 2025-02-11 09:15:00 | 430.20 | EXIT_EMA400 | -41.45 |
| SELL | 2025-06-19 11:15:00 | 449.05 | 2025-06-23 10:15:00 | 421.89 | TARGET | 27.16 |
| SELL | 2025-05-26 09:15:00 | 439.45 | 2025-07-11 09:15:00 | 460.70 | EXIT_EMA400 | -21.25 |
| BUY | 2025-09-19 09:15:00 | 575.75 | 2025-09-30 11:15:00 | 547.30 | EXIT_EMA400 | -28.45 |
| SELL | 2026-01-08 10:15:00 | 493.10 | 2026-02-03 12:15:00 | 501.00 | EXIT_EMA400 | -7.90 |
| BUY | 2026-03-18 10:15:00 | 533.55 | 2026-03-19 10:15:00 | 516.35 | EXIT_EMA400 | -17.20 |
