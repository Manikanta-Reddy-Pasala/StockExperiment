# Krishna Institute of Medical Sciences Ltd. (KIMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 666.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 1
- **Winners / losers:** 4 / 5
- **Target hits / EMA400 exits:** 4 / 5
- **Total realized P&L (per unit):** 67.40
- **Avg P&L per closed trade:** 7.49

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-02 13:15:00 | 371.25 | 384.59 | 384.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-02 14:15:00 | 369.00 | 384.43 | 384.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-20 09:15:00 | 379.45 | 379.31 | 381.52 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 389.80 | 383.19 | 383.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 394.00 | 383.36 | 383.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-08 13:15:00 | 382.72 | 386.31 | 384.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 14:15:00 | 387.00 | 385.21 | 384.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 387.00 | 385.21 | 384.47 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-15 09:15:00 | 388.78 | 385.27 | 384.51 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 388.98 | 391.19 | 388.32 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-02 15:15:00 | 388.03 | 391.16 | 388.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-04-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 13:15:00 | 409.02 | 414.71 | 414.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-05 15:15:00 | 407.22 | 414.59 | 414.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 12:15:00 | 405.63 | 405.38 | 409.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-26 09:15:00 | 399.91 | 405.43 | 409.07 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 403.50 | 404.90 | 408.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-02 14:15:00 | 396.84 | 404.69 | 408.23 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-06 10:15:00 | 409.42 | 404.29 | 407.85 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 427.78 | 397.42 | 397.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 429.69 | 399.53 | 398.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 09:15:00 | 422.35 | 422.45 | 414.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-06 09:15:00 | 429.59 | 422.12 | 415.06 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-25 10:15:00 | 515.45 | 536.05 | 515.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 543.25 | 595.59 | 595.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 12:15:00 | 540.15 | 595.04 | 595.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 14:15:00 | 570.00 | 550.87 | 567.90 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 615.45 | 579.28 | 579.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 13:15:00 | 632.35 | 581.05 | 580.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 13:15:00 | 586.65 | 588.68 | 584.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-04 14:15:00 | 593.10 | 588.72 | 584.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 593.10 | 588.72 | 584.45 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 559.45 | 588.49 | 584.37 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 700.05 | 722.66 | 722.75 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 727.45 | 722.72 | 722.70 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 720.20 | 722.68 | 722.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 718.25 | 722.64 | 722.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 725.00 | 722.59 | 722.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-24 15:15:00 | 719.90 | 722.56 | 722.62 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-27 09:15:00 | 722.65 | 722.56 | 722.62 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 735.35 | 722.77 | 722.72 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 708.80 | 722.62 | 722.67 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 727.65 | 722.73 | 722.72 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 704.50 | 722.58 | 722.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 698.80 | 722.34 | 722.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 700.25 | 691.72 | 703.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 14:15:00 | 678.35 | 693.82 | 702.23 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-09 11:15:00 | 646.50 | 617.27 | 637.79 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 697.90 | 651.84 | 651.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 710.70 | 653.67 | 652.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 683.75 | 684.96 | 672.31 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 634.45 | 664.10 | 664.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 619.90 | 662.56 | 663.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 671.80 | 652.10 | 657.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-13 09:15:00 | 650.65 | 654.22 | 658.13 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 660.40 | 654.28 | 658.14 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 680.00 | 661.22 | 661.16 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 647.35 | 660.99 | 661.04 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 669.20 | 661.07 | 661.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 12:15:00 | 671.30 | 661.17 | 661.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 659.55 | 661.27 | 661.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-04-30 14:15:00 | 667.30 | 661.48 | 661.25 | Buy entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-14 14:15:00 | 387.00 | 2023-12-15 12:15:00 | 394.58 | TARGET | 7.58 |
| BUY | 2023-12-15 09:15:00 | 388.78 | 2023-12-18 09:15:00 | 401.60 | TARGET | 12.82 |
| SELL | 2024-04-26 09:15:00 | 399.91 | 2024-05-06 10:15:00 | 409.42 | EXIT_EMA400 | -9.51 |
| SELL | 2024-05-02 14:15:00 | 396.84 | 2024-05-06 10:15:00 | 409.42 | EXIT_EMA400 | -12.58 |
| BUY | 2024-08-06 09:15:00 | 429.59 | 2024-08-19 11:15:00 | 473.18 | TARGET | 43.59 |
| BUY | 2025-04-04 14:15:00 | 593.10 | 2025-04-07 09:15:00 | 559.45 | EXIT_EMA400 | -33.65 |
| SELL | 2025-10-24 15:15:00 | 719.90 | 2025-10-27 09:15:00 | 722.65 | EXIT_EMA400 | -2.75 |
| SELL | 2025-12-08 14:15:00 | 678.35 | 2025-12-30 09:15:00 | 606.70 | TARGET | 71.65 |
| SELL | 2026-04-13 09:15:00 | 650.65 | 2026-04-13 10:15:00 | 660.40 | EXIT_EMA400 | -9.75 |
