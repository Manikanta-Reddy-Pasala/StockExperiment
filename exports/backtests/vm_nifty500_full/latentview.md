# Latent View Analytics Ltd. (LATENTVIEW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 292.35
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 13 |
| ALERT2 | 13 |
| ALERT3 | 6 |
| ENTRY1 | 9 |
| ENTRY2 | 2 |
| EXIT | 8 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 4 / 7
- **Target hits / EMA400 exits:** 4 / 7
- **Total realized P&L (per unit):** 108.09
- **Avg P&L per closed trade:** 9.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 456.85 | 478.15 | 478.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 13:15:00 | 454.10 | 477.91 | 478.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 10:15:00 | 483.00 | 476.46 | 477.31 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 516.15 | 478.27 | 478.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-01 11:15:00 | 526.45 | 478.75 | 478.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 09:15:00 | 510.90 | 511.86 | 498.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-19 11:15:00 | 512.20 | 511.80 | 498.75 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 500.00 | 511.65 | 499.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-23 13:15:00 | 498.90 | 511.30 | 499.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 470.00 | 493.96 | 494.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-15 10:15:00 | 464.00 | 493.17 | 493.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 10:15:00 | 492.00 | 488.80 | 491.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-27 12:15:00 | 478.50 | 489.12 | 491.15 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 482.00 | 485.74 | 489.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-04 09:15:00 | 466.00 | 484.65 | 488.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-06 14:15:00 | 496.10 | 482.02 | 486.60 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 14:15:00 | 512.50 | 490.56 | 490.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 15:15:00 | 515.00 | 490.80 | 490.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 13:15:00 | 521.00 | 521.16 | 511.14 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 500.30 | 506.05 | 506.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-16 15:15:00 | 493.00 | 505.72 | 505.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 507.05 | 505.13 | 505.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-26 14:15:00 | 496.05 | 505.11 | 505.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 489.15 | 490.37 | 496.58 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-12 14:15:00 | 487.60 | 490.37 | 496.49 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-09-13 09:15:00 | 497.50 | 490.45 | 496.47 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 11:15:00 | 501.90 | 472.14 | 472.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 504.00 | 472.74 | 472.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 474.20 | 478.37 | 475.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-27 09:15:00 | 480.50 | 477.02 | 475.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 480.50 | 477.02 | 475.14 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-30 10:15:00 | 475.30 | 477.40 | 475.41 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 15:15:00 | 444.85 | 475.44 | 475.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 440.95 | 466.38 | 470.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 15:15:00 | 453.00 | 451.07 | 460.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-07 14:15:00 | 447.85 | 452.45 | 460.41 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 13:15:00 | 393.35 | 376.48 | 393.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-17 14:15:00 | 394.85 | 376.66 | 393.78 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 422.00 | 401.13 | 401.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 425.85 | 406.24 | 403.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 10:15:00 | 409.75 | 410.10 | 406.53 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 12:15:00 | 412.00 | 410.15 | 406.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 404.85 | 411.01 | 407.74 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 10:15:00 | 389.80 | 415.94 | 415.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-11 11:15:00 | 388.75 | 415.66 | 415.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 428.90 | 410.56 | 412.94 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 425.35 | 415.02 | 414.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-28 09:15:00 | 436.40 | 415.35 | 415.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-29 15:15:00 | 412.85 | 415.99 | 415.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-03 12:15:00 | 423.90 | 415.92 | 415.51 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-12 13:15:00 | 417.10 | 419.68 | 417.76 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 14:15:00 | 406.30 | 416.73 | 416.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 15:15:00 | 404.80 | 416.61 | 416.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-03 14:15:00 | 415.05 | 414.71 | 415.64 | EMA200 retest candle locked |

### Cycle 12 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 435.00 | 416.61 | 416.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 441.10 | 418.07 | 417.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 418.65 | 424.64 | 421.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-31 10:15:00 | 437.00 | 423.65 | 421.23 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 454.40 | 468.05 | 453.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-05 14:15:00 | 451.00 | 467.88 | 453.63 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 434.40 | 458.86 | 458.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 428.65 | 458.56 | 458.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 444.85 | 432.56 | 443.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 09:15:00 | 414.40 | 434.18 | 442.67 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-19 11:15:00 | 512.20 | 2024-04-23 13:15:00 | 498.90 | EXIT_EMA400 | -13.30 |
| SELL | 2024-05-27 12:15:00 | 478.50 | 2024-06-06 14:15:00 | 496.10 | EXIT_EMA400 | -17.60 |
| SELL | 2024-06-04 09:15:00 | 466.00 | 2024-06-06 14:15:00 | 496.10 | EXIT_EMA400 | -30.10 |
| SELL | 2024-08-26 14:15:00 | 496.05 | 2024-09-09 09:15:00 | 467.68 | TARGET | 28.37 |
| SELL | 2024-09-12 14:15:00 | 487.60 | 2024-09-13 09:15:00 | 497.50 | EXIT_EMA400 | -9.90 |
| BUY | 2024-12-27 09:15:00 | 480.50 | 2024-12-30 10:15:00 | 475.30 | EXIT_EMA400 | -5.20 |
| SELL | 2025-02-07 14:15:00 | 447.85 | 2025-02-11 12:15:00 | 410.17 | TARGET | 37.68 |
| BUY | 2025-06-06 12:15:00 | 412.00 | 2025-06-12 13:15:00 | 404.85 | EXIT_EMA400 | -7.15 |
| BUY | 2025-09-03 12:15:00 | 423.90 | 2025-09-12 13:15:00 | 417.10 | EXIT_EMA400 | -6.80 |
| BUY | 2025-10-31 10:15:00 | 437.00 | 2025-11-11 11:15:00 | 484.30 | TARGET | 47.30 |
| SELL | 2026-02-10 09:15:00 | 414.40 | 2026-03-02 09:15:00 | 329.60 | TARGET | 84.80 |
