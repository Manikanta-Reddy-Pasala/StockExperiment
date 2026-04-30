# H.E.G. Ltd. (HEG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 596.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 14 |
| ALERT2 | 14 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 0 |
| EXIT | 7 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -83.34
- **Avg P&L per closed trade:** -11.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 12:15:00 | 331.40 | 338.16 | 338.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-06 15:15:00 | 329.01 | 337.92 | 338.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 326.80 | 325.52 | 330.47 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-12 10:15:00 | 347.02 | 333.77 | 333.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 350.36 | 335.48 | 334.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 10:15:00 | 358.81 | 360.32 | 350.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-10 11:15:00 | 373.26 | 360.45 | 350.92 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 358.80 | 363.24 | 354.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-23 14:15:00 | 349.39 | 362.59 | 354.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 326.77 | 353.15 | 353.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 322.91 | 349.20 | 351.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-12 09:15:00 | 345.80 | 342.90 | 347.09 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 13:15:00 | 358.53 | 350.30 | 350.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 383.20 | 350.83 | 350.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 14:15:00 | 445.79 | 471.33 | 440.81 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 13:15:00 | 414.75 | 439.82 | 439.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 407.19 | 439.04 | 439.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 10:15:00 | 453.90 | 431.52 | 435.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-05 09:15:00 | 422.68 | 433.79 | 435.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-13 09:15:00 | 432.07 | 426.81 | 431.55 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 13:15:00 | 464.27 | 422.26 | 422.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 09:15:00 | 499.31 | 423.79 | 422.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-07 10:15:00 | 447.11 | 447.21 | 436.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-08 09:15:00 | 460.48 | 447.32 | 436.73 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-22 09:15:00 | 445.80 | 469.60 | 452.98 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 14:15:00 | 405.10 | 443.60 | 443.71 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 12:15:00 | 563.35 | 440.76 | 440.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 14:15:00 | 576.00 | 443.27 | 441.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 13:15:00 | 514.80 | 516.96 | 489.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-30 09:15:00 | 519.70 | 516.65 | 490.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-01-06 13:15:00 | 493.90 | 517.10 | 495.56 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 12:15:00 | 431.95 | 481.42 | 481.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 09:15:00 | 427.60 | 479.44 | 480.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-06 09:15:00 | 388.15 | 374.20 | 403.92 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 479.50 | 415.63 | 415.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 483.95 | 417.50 | 416.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 456.80 | 459.49 | 445.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-14 09:15:00 | 475.80 | 455.97 | 446.29 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-19 15:15:00 | 483.00 | 500.22 | 483.77 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 14:15:00 | 461.20 | 505.42 | 505.48 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 519.20 | 504.11 | 504.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 11:15:00 | 525.30 | 505.12 | 504.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 506.65 | 510.37 | 507.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 514.65 | 510.35 | 507.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 514.65 | 510.35 | 507.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-30 10:15:00 | 506.25 | 510.23 | 507.71 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 530.90 | 549.97 | 550.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 14:15:00 | 524.40 | 548.32 | 549.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 10:15:00 | 551.20 | 548.10 | 549.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-19 14:15:00 | 545.50 | 548.49 | 549.19 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 545.50 | 548.49 | 549.19 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-20 09:15:00 | 553.90 | 548.51 | 549.20 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 561.60 | 549.86 | 549.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 564.65 | 550.01 | 549.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 553.85 | 557.07 | 553.74 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 10:15:00 | 516.60 | 550.80 | 550.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 504.40 | 545.76 | 548.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 09:15:00 | 564.00 | 522.64 | 534.18 | EMA200 retest candle locked |

### Cycle 16 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 590.70 | 541.79 | 541.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 607.40 | 544.32 | 543.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 588.95 | 597.38 | 574.08 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-10 11:15:00 | 373.26 | 2024-01-23 14:15:00 | 349.39 | EXIT_EMA400 | -23.87 |
| SELL | 2024-08-05 09:15:00 | 422.68 | 2024-08-13 09:15:00 | 432.07 | EXIT_EMA400 | -9.39 |
| BUY | 2024-10-08 09:15:00 | 460.48 | 2024-10-22 09:15:00 | 445.80 | EXIT_EMA400 | -14.68 |
| BUY | 2024-12-30 09:15:00 | 519.70 | 2025-01-06 13:15:00 | 493.90 | EXIT_EMA400 | -25.80 |
| BUY | 2025-05-14 09:15:00 | 475.80 | 2025-06-19 15:15:00 | 483.00 | EXIT_EMA400 | 7.20 |
| BUY | 2025-09-29 09:15:00 | 514.65 | 2025-09-30 10:15:00 | 506.25 | EXIT_EMA400 | -8.40 |
| SELL | 2026-02-19 14:15:00 | 545.50 | 2026-02-20 09:15:00 | 553.90 | EXIT_EMA400 | -8.40 |
