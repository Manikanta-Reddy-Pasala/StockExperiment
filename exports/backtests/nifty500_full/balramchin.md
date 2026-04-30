# Balrampur Chini Mills Ltd. (BALRAMCHIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 521.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -97.74
- **Avg P&L per closed trade:** -10.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-28 15:15:00 | 389.60 | 392.96 | 392.97 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 14:15:00 | 395.40 | 393.00 | 392.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 401.40 | 393.11 | 393.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-12 14:15:00 | 399.30 | 399.60 | 396.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-13 09:15:00 | 404.60 | 399.66 | 396.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-23 09:15:00 | 411.45 | 422.06 | 415.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 10:15:00 | 404.20 | 424.75 | 424.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-18 14:15:00 | 401.95 | 423.92 | 424.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 11:15:00 | 414.05 | 411.71 | 417.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-02 09:15:00 | 405.50 | 411.96 | 417.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 13:15:00 | 403.10 | 395.91 | 403.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-06 14:15:00 | 396.75 | 396.08 | 403.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-02-07 10:15:00 | 404.90 | 396.19 | 403.29 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 15:15:00 | 392.20 | 382.82 | 382.80 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-08 11:15:00 | 380.35 | 382.78 | 382.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 10:15:00 | 376.05 | 382.59 | 382.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 380.80 | 380.58 | 381.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-21 09:15:00 | 376.90 | 380.57 | 381.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 376.90 | 380.57 | 381.50 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-05-24 10:15:00 | 381.35 | 380.14 | 381.17 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 12:15:00 | 404.25 | 381.79 | 381.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 13:15:00 | 405.10 | 382.02 | 381.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 609.55 | 622.83 | 583.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-29 11:15:00 | 632.70 | 621.02 | 588.76 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-07 15:15:00 | 594.90 | 620.74 | 595.46 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 559.00 | 579.33 | 579.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 552.40 | 577.85 | 578.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 578.00 | 577.03 | 578.19 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 605.70 | 579.23 | 579.22 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 530.15 | 579.59 | 579.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 527.45 | 579.07 | 579.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 13:15:00 | 470.15 | 469.00 | 497.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-25 12:15:00 | 457.35 | 469.53 | 494.66 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-12 14:15:00 | 489.95 | 461.15 | 481.86 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 546.60 | 494.43 | 494.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 15:15:00 | 551.35 | 495.00 | 494.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 545.55 | 545.92 | 528.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 562.15 | 546.39 | 529.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 539.05 | 547.38 | 531.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 11:15:00 | 528.05 | 547.04 | 531.11 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 550.00 | 581.57 | 581.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 09:15:00 | 547.20 | 579.99 | 580.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 576.20 | 574.85 | 577.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 10:15:00 | 570.25 | 577.38 | 578.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 577.70 | 570.80 | 575.22 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 461.65 | 445.39 | 445.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 14:15:00 | 465.50 | 445.59 | 445.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-17 12:15:00 | 469.30 | 470.99 | 460.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 09:15:00 | 482.85 | 471.05 | 460.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-24 09:15:00 | 452.00 | 472.87 | 463.30 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-13 09:15:00 | 404.60 | 2023-09-14 09:15:00 | 428.25 | TARGET | 23.65 |
| SELL | 2024-01-02 09:15:00 | 405.50 | 2024-02-07 10:15:00 | 404.90 | EXIT_EMA400 | 0.60 |
| SELL | 2024-02-06 14:15:00 | 396.75 | 2024-02-07 10:15:00 | 404.90 | EXIT_EMA400 | -8.15 |
| SELL | 2024-05-21 09:15:00 | 376.90 | 2024-05-24 10:15:00 | 381.35 | EXIT_EMA400 | -4.45 |
| BUY | 2024-10-29 11:15:00 | 632.70 | 2024-11-07 15:15:00 | 594.90 | EXIT_EMA400 | -37.80 |
| SELL | 2025-02-25 12:15:00 | 457.35 | 2025-03-12 14:15:00 | 489.95 | EXIT_EMA400 | -32.60 |
| BUY | 2025-05-07 10:15:00 | 562.15 | 2025-05-09 11:15:00 | 528.05 | EXIT_EMA400 | -34.10 |
| SELL | 2025-08-26 10:15:00 | 570.25 | 2025-08-28 10:15:00 | 544.29 | TARGET | 25.96 |
| BUY | 2026-03-18 09:15:00 | 482.85 | 2026-03-24 09:15:00 | 452.00 | EXIT_EMA400 | -30.85 |
