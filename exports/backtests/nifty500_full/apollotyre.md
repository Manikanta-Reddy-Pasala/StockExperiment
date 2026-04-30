# Apollo Tyres Ltd. (APOLLOTYRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 408.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 17 |
| ALERT1 | 17 |
| ALERT2 | 17 |
| ALERT3 | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 9 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 5 / 7
- **Target hits / EMA400 exits:** 5 / 7
- **Total realized P&L (per unit):** 14.54
- **Avg P&L per closed trade:** 1.21

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 10:15:00 | 392.00 | 408.65 | 408.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 14:15:00 | 389.70 | 407.01 | 407.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 12:15:00 | 381.95 | 381.09 | 389.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-09 09:15:00 | 370.00 | 380.96 | 389.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-10-17 09:15:00 | 391.60 | 380.01 | 387.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 11:15:00 | 414.65 | 388.56 | 388.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-15 09:15:00 | 421.25 | 389.93 | 389.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 14:15:00 | 431.05 | 436.05 | 420.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-29 09:15:00 | 443.70 | 435.49 | 422.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 500.90 | 517.35 | 499.62 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 10:15:00 | 497.45 | 517.15 | 499.61 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 13:15:00 | 469.60 | 488.49 | 488.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 466.70 | 486.64 | 487.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 10:15:00 | 483.00 | 482.01 | 484.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-10 11:15:00 | 478.65 | 481.98 | 484.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-10 13:15:00 | 485.20 | 482.01 | 484.94 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 09:15:00 | 513.70 | 485.95 | 485.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 10:15:00 | 515.85 | 486.24 | 486.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-06 10:15:00 | 483.75 | 488.66 | 487.32 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-05-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 12:15:00 | 478.15 | 486.18 | 486.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 09:15:00 | 477.00 | 485.92 | 486.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 09:15:00 | 503.15 | 484.01 | 485.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 10:15:00 | 488.30 | 484.05 | 485.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 10:15:00 | 488.30 | 484.05 | 485.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-16 11:15:00 | 490.10 | 484.11 | 485.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 498.65 | 482.80 | 482.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 12:15:00 | 502.85 | 483.18 | 482.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 517.90 | 518.53 | 506.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-24 09:15:00 | 532.75 | 518.71 | 506.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-06 14:15:00 | 513.90 | 530.46 | 517.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 496.80 | 509.22 | 509.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 11:15:00 | 493.90 | 508.94 | 509.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 508.25 | 507.03 | 508.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 09:15:00 | 503.45 | 507.39 | 508.23 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 503.45 | 507.39 | 508.23 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-06 14:15:00 | 510.65 | 507.30 | 508.16 | Close above EMA400 |

### Cycle 8 — BUY (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 15:15:00 | 521.00 | 508.95 | 508.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 13:15:00 | 526.20 | 510.34 | 509.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 512.05 | 512.36 | 510.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-19 14:15:00 | 518.65 | 512.30 | 510.82 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-04 09:15:00 | 516.40 | 524.17 | 517.96 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 502.40 | 514.12 | 514.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 495.35 | 513.68 | 513.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 12:15:00 | 492.55 | 491.55 | 499.60 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 541.45 | 504.97 | 504.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 543.00 | 505.72 | 505.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 09:15:00 | 523.65 | 527.08 | 519.31 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 461.45 | 514.21 | 514.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 452.65 | 513.08 | 513.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 413.00 | 411.37 | 433.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 387.50 | 418.47 | 429.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-11 14:15:00 | 427.75 | 415.30 | 426.61 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 15:15:00 | 461.00 | 434.63 | 434.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 461.40 | 436.38 | 435.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 12:15:00 | 475.55 | 475.82 | 462.30 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 447.65 | 458.11 | 458.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 15:15:00 | 445.95 | 457.99 | 458.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 462.10 | 456.80 | 457.46 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 464.95 | 458.13 | 458.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 470.00 | 458.25 | 458.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 459.40 | 460.38 | 459.28 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 448.60 | 458.51 | 458.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 446.90 | 457.11 | 457.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 10:15:00 | 456.10 | 448.04 | 452.21 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 467.75 | 455.32 | 455.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 485.00 | 455.74 | 455.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 12:15:00 | 475.60 | 476.69 | 469.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-26 13:15:00 | 480.50 | 476.73 | 469.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 467.50 | 476.60 | 469.35 | Close below EMA400 |

### Cycle 17 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 484.25 | 505.48 | 505.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 480.95 | 505.23 | 505.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 14:15:00 | 504.20 | 503.83 | 504.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 10:15:00 | 501.60 | 505.31 | 505.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 501.60 | 505.31 | 505.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 11:15:00 | 500.00 | 505.26 | 505.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 447.25 | 432.11 | 448.20 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-17 12:15:00 | 444.70 | 432.39 | 448.18 | Sell entry 2 (retest2 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-09 09:15:00 | 370.00 | 2023-10-17 09:15:00 | 391.60 | EXIT_EMA400 | -21.60 |
| BUY | 2023-12-29 09:15:00 | 443.70 | 2024-01-23 09:15:00 | 506.34 | TARGET | 62.64 |
| SELL | 2024-04-10 11:15:00 | 478.65 | 2024-04-10 13:15:00 | 485.20 | EXIT_EMA400 | -6.55 |
| SELL | 2024-05-16 10:15:00 | 488.30 | 2024-05-16 11:15:00 | 490.10 | EXIT_EMA400 | -1.80 |
| BUY | 2024-07-24 09:15:00 | 532.75 | 2024-08-06 14:15:00 | 513.90 | EXIT_EMA400 | -18.85 |
| SELL | 2024-09-06 09:15:00 | 503.45 | 2024-09-06 14:15:00 | 510.65 | EXIT_EMA400 | -7.20 |
| BUY | 2024-09-19 14:15:00 | 518.65 | 2024-09-25 09:15:00 | 542.14 | TARGET | 23.49 |
| SELL | 2025-04-07 09:15:00 | 387.50 | 2025-04-11 14:15:00 | 427.75 | EXIT_EMA400 | -40.25 |
| BUY | 2025-09-26 13:15:00 | 480.50 | 2025-09-29 11:15:00 | 467.50 | EXIT_EMA400 | -13.00 |
| SELL | 2026-02-12 10:15:00 | 501.60 | 2026-02-13 13:15:00 | 490.35 | TARGET | 11.25 |
| SELL | 2026-02-12 11:15:00 | 500.00 | 2026-02-16 09:15:00 | 484.03 | TARGET | 15.97 |
| SELL | 2026-04-17 12:15:00 | 444.70 | 2026-04-22 14:15:00 | 434.25 | TARGET | 10.45 |
