# Sumitomo Chemical India Ltd. (SUMICHEM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 419.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 10 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 7 / 6
- **Target hits / EMA400 exits:** 7 / 6
- **Total realized P&L (per unit):** 131.06
- **Avg P&L per closed trade:** 10.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 405.50 | 412.03 | 412.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 404.95 | 411.96 | 412.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 421.05 | 411.37 | 411.69 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 433.90 | 412.22 | 412.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 435.75 | 412.64 | 412.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 14:15:00 | 427.95 | 428.21 | 421.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-05 11:15:00 | 439.50 | 425.44 | 422.54 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-10 09:15:00 | 428.25 | 426.87 | 423.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-10-10 11:15:00 | 423.50 | 426.85 | 423.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 13:15:00 | 385.00 | 421.73 | 421.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 374.50 | 420.52 | 421.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 09:15:00 | 393.45 | 392.11 | 400.80 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-02 11:15:00 | 418.90 | 405.29 | 405.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-08 09:15:00 | 426.80 | 407.15 | 406.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 14:15:00 | 408.20 | 408.64 | 407.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 412.60 | 408.68 | 407.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-11 13:15:00 | 406.50 | 408.72 | 407.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-24 11:15:00 | 394.45 | 406.14 | 406.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-07 13:15:00 | 391.70 | 403.08 | 404.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 14:15:00 | 371.85 | 364.10 | 377.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-27 09:15:00 | 348.00 | 363.31 | 376.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 370.15 | 361.34 | 373.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-03 14:15:00 | 373.65 | 361.46 | 373.07 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 11:15:00 | 405.90 | 380.14 | 380.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 15:15:00 | 407.05 | 383.83 | 381.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 11:15:00 | 389.95 | 390.55 | 386.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 11:15:00 | 400.40 | 389.92 | 386.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 523.35 | 520.49 | 503.64 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-09-06 09:15:00 | 531.75 | 520.46 | 504.76 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-07 09:15:00 | 532.60 | 554.48 | 536.19 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 519.30 | 538.20 | 538.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 513.25 | 536.87 | 537.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-30 11:15:00 | 542.30 | 531.68 | 534.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 10:15:00 | 522.70 | 532.46 | 534.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 500.05 | 502.79 | 515.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-28 11:15:00 | 495.10 | 502.67 | 515.04 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 515.90 | 502.60 | 514.28 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 540.00 | 503.85 | 503.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 12:15:00 | 542.95 | 504.96 | 504.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 12:15:00 | 537.25 | 537.65 | 525.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-16 15:15:00 | 541.05 | 524.53 | 521.89 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 526.30 | 527.54 | 523.99 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-27 09:15:00 | 506.95 | 527.48 | 524.08 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 504.05 | 521.14 | 521.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 502.60 | 520.95 | 521.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 14:15:00 | 519.05 | 518.52 | 519.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-09 10:15:00 | 515.45 | 518.52 | 519.75 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 515.45 | 518.52 | 519.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-09 11:15:00 | 514.80 | 518.48 | 519.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 519.00 | 518.46 | 519.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-10 11:15:00 | 512.10 | 518.31 | 519.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 512.95 | 509.65 | 514.21 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-06-24 09:15:00 | 516.70 | 509.74 | 514.22 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 12:15:00 | 553.90 | 517.06 | 517.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 561.55 | 518.56 | 517.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 583.55 | 585.65 | 561.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-12 15:15:00 | 594.50 | 585.33 | 563.82 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 572.65 | 587.51 | 570.82 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-26 15:15:00 | 564.10 | 587.28 | 570.79 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 11:15:00 | 531.25 | 566.64 | 566.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 13:15:00 | 530.95 | 565.97 | 566.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 462.60 | 462.17 | 481.44 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-07 13:15:00 | 459.75 | 464.90 | 478.25 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 406.60 | 391.02 | 406.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 407.55 | 391.34 | 406.89 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 443.25 | 416.14 | 416.08 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-05 11:15:00 | 439.50 | 2023-10-10 11:15:00 | 423.50 | EXIT_EMA400 | -16.00 |
| BUY | 2024-01-11 09:15:00 | 412.60 | 2024-01-11 13:15:00 | 406.50 | EXIT_EMA400 | -6.10 |
| SELL | 2024-03-27 09:15:00 | 348.00 | 2024-04-03 14:15:00 | 373.65 | EXIT_EMA400 | -25.65 |
| BUY | 2024-05-14 11:15:00 | 400.40 | 2024-05-23 09:15:00 | 442.27 | TARGET | 41.87 |
| BUY | 2024-09-06 09:15:00 | 531.75 | 2024-09-10 14:15:00 | 612.71 | TARGET | 80.96 |
| SELL | 2025-01-06 10:15:00 | 522.70 | 2025-01-10 09:15:00 | 487.12 | TARGET | 35.58 |
| SELL | 2025-01-28 11:15:00 | 495.10 | 2025-01-30 09:15:00 | 515.90 | EXIT_EMA400 | -20.80 |
| BUY | 2025-05-16 15:15:00 | 541.05 | 2025-05-27 09:15:00 | 506.95 | EXIT_EMA400 | -34.10 |
| SELL | 2025-06-09 10:15:00 | 515.45 | 2025-06-12 10:15:00 | 502.54 | TARGET | 12.91 |
| SELL | 2025-06-09 11:15:00 | 514.80 | 2025-06-12 12:15:00 | 500.02 | TARGET | 14.78 |
| SELL | 2025-06-10 11:15:00 | 512.10 | 2025-06-13 09:15:00 | 489.61 | TARGET | 22.49 |
| BUY | 2025-08-12 15:15:00 | 594.50 | 2025-08-26 15:15:00 | 564.10 | EXIT_EMA400 | -30.40 |
| SELL | 2026-01-07 13:15:00 | 459.75 | 2026-01-27 14:15:00 | 404.24 | TARGET | 55.51 |
