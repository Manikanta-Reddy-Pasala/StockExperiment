# Star Health and Allied Insurance Company Ltd. (STARHEALTH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 525.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT3 | 10 |
| ENTRY1 | 9 |
| ENTRY2 | 4 |
| EXIT | 9 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 4 / 9
- **Target hits / EMA400 exits:** 4 / 9
- **Total realized P&L (per unit):** 59.91
- **Avg P&L per closed trade:** 4.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 12:15:00 | 587.35 | 612.37 | 612.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 14:15:00 | 581.75 | 611.81 | 612.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 12:15:00 | 566.05 | 565.33 | 579.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-28 15:15:00 | 560.00 | 566.04 | 579.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-30 14:15:00 | 579.90 | 566.48 | 578.44 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 15:15:00 | 572.65 | 561.70 | 561.66 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 10:15:00 | 558.20 | 561.61 | 561.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 15:15:00 | 551.90 | 561.29 | 561.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-19 09:15:00 | 561.50 | 558.76 | 560.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-22 09:15:00 | 555.00 | 559.99 | 560.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 555.00 | 559.99 | 560.59 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-23 11:15:00 | 561.45 | 559.85 | 560.49 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 578.70 | 561.21 | 561.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 14:15:00 | 580.00 | 562.09 | 561.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-29 09:15:00 | 562.30 | 562.81 | 561.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 547.35 | 561.21 | 561.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 540.50 | 558.52 | 559.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-15 14:15:00 | 557.75 | 556.41 | 558.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-19 09:15:00 | 526.80 | 555.81 | 558.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 554.65 | 551.51 | 555.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-28 10:15:00 | 547.05 | 551.46 | 555.30 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 554.90 | 551.12 | 555.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 10:15:00 | 555.05 | 551.16 | 555.01 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 12:15:00 | 576.10 | 557.37 | 557.30 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 13:15:00 | 533.60 | 557.91 | 557.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 12:15:00 | 529.70 | 555.24 | 556.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 12:15:00 | 551.60 | 550.29 | 553.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-21 10:15:00 | 549.15 | 550.38 | 553.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 551.55 | 549.35 | 552.86 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-23 11:15:00 | 548.00 | 549.37 | 552.83 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 545.00 | 549.27 | 552.69 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-27 09:15:00 | 534.50 | 548.87 | 552.37 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-28 09:15:00 | 549.85 | 526.33 | 534.31 | Close above EMA400 |

### Cycle 8 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 580.00 | 540.89 | 540.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 14:15:00 | 585.15 | 542.09 | 541.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 571.35 | 583.98 | 568.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-05 11:15:00 | 581.80 | 583.83 | 568.64 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 569.05 | 582.85 | 569.02 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-07 12:15:00 | 577.80 | 582.58 | 569.10 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 571.55 | 582.42 | 571.02 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-14 10:15:00 | 569.65 | 582.29 | 571.01 | Close below EMA400 |

### Cycle 9 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 550.95 | 592.39 | 592.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 548.50 | 585.77 | 588.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 495.05 | 491.12 | 520.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 471.10 | 490.89 | 517.60 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-16 09:15:00 | 389.70 | 363.63 | 387.14 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 437.20 | 392.43 | 392.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 09:15:00 | 447.80 | 394.31 | 393.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 438.85 | 442.78 | 425.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-16 11:15:00 | 451.65 | 442.85 | 425.38 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 426.70 | 442.01 | 426.69 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 425.05 | 441.84 | 426.68 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 12:15:00 | 459.25 | 473.29 | 473.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 455.30 | 472.46 | 472.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 09:15:00 | 469.30 | 465.32 | 468.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-02 12:15:00 | 461.00 | 465.33 | 468.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 455.10 | 448.31 | 456.89 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-30 09:15:00 | 469.75 | 448.72 | 456.80 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 15:15:00 | 478.40 | 461.75 | 461.74 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 451.70 | 461.80 | 461.81 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2026-02-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 13:15:00 | 469.80 | 461.88 | 461.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 09:15:00 | 475.50 | 462.15 | 461.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 460.60 | 463.13 | 462.51 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 444.80 | 461.85 | 461.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 10:15:00 | 440.55 | 461.64 | 461.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 463.00 | 459.68 | 460.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 10:15:00 | 458.60 | 459.71 | 460.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 458.60 | 459.71 | 460.73 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-13 14:15:00 | 464.50 | 459.77 | 460.74 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 470.05 | 460.47 | 460.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 15:15:00 | 475.00 | 460.87 | 460.63 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-28 15:15:00 | 560.00 | 2023-11-30 14:15:00 | 579.90 | EXIT_EMA400 | -19.90 |
| SELL | 2024-02-22 09:15:00 | 555.00 | 2024-02-23 11:15:00 | 561.45 | EXIT_EMA400 | -6.45 |
| SELL | 2024-03-19 09:15:00 | 526.80 | 2024-04-01 10:15:00 | 555.05 | EXIT_EMA400 | -28.25 |
| SELL | 2024-03-28 10:15:00 | 547.05 | 2024-04-01 10:15:00 | 555.05 | EXIT_EMA400 | -8.00 |
| SELL | 2024-05-21 10:15:00 | 549.15 | 2024-05-27 09:15:00 | 535.84 | TARGET | 13.31 |
| SELL | 2024-05-23 11:15:00 | 548.00 | 2024-05-27 09:15:00 | 533.51 | TARGET | 14.49 |
| SELL | 2024-05-27 09:15:00 | 534.50 | 2024-06-28 09:15:00 | 549.85 | EXIT_EMA400 | -15.35 |
| BUY | 2024-08-05 11:15:00 | 581.80 | 2024-08-14 10:15:00 | 569.65 | EXIT_EMA400 | -12.15 |
| BUY | 2024-08-07 12:15:00 | 577.80 | 2024-08-14 10:15:00 | 569.65 | EXIT_EMA400 | -8.15 |
| SELL | 2024-12-09 09:15:00 | 471.10 | 2025-04-07 09:15:00 | 331.59 | TARGET | 139.51 |
| BUY | 2025-06-16 11:15:00 | 451.65 | 2025-06-19 12:15:00 | 425.05 | EXIT_EMA400 | -26.60 |
| SELL | 2026-01-02 12:15:00 | 461.00 | 2026-01-19 10:15:00 | 437.64 | TARGET | 23.36 |
| SELL | 2026-03-13 10:15:00 | 458.60 | 2026-03-13 14:15:00 | 464.50 | EXIT_EMA400 | -5.90 |
