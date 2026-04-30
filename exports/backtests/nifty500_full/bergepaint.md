# Berger Paints India Ltd. (BERGEPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 473.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 8 |
| ENTRY1 | 8 |
| ENTRY2 | 4 |
| EXIT | 8 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 8 / 4
- **Target hits / EMA400 exits:** 8 / 4
- **Total realized P&L (per unit):** 139.23
- **Avg P&L per closed trade:** 11.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 10:15:00 | 572.45 | 581.38 | 581.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 11:15:00 | 569.00 | 581.26 | 581.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 11:15:00 | 581.00 | 580.07 | 580.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-23 09:15:00 | 574.10 | 580.13 | 580.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 574.10 | 580.13 | 580.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-10-23 14:15:00 | 569.70 | 579.84 | 580.56 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-06 11:15:00 | 573.60 | 567.45 | 573.24 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 15:15:00 | 586.95 | 575.68 | 575.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 591.00 | 576.33 | 576.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-11 11:15:00 | 575.00 | 577.20 | 576.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 14:15:00 | 580.30 | 576.32 | 576.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 580.30 | 576.32 | 576.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-14 15:15:00 | 581.00 | 576.37 | 576.13 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-20 13:15:00 | 577.60 | 579.52 | 577.82 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 12:15:00 | 557.00 | 581.52 | 581.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 10:15:00 | 551.75 | 573.56 | 577.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 11:15:00 | 568.75 | 565.66 | 571.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-22 09:15:00 | 563.45 | 565.85 | 571.43 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 563.45 | 565.85 | 571.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-02-22 14:15:00 | 571.55 | 566.01 | 571.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 525.55 | 512.54 | 512.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 10:15:00 | 529.70 | 512.96 | 512.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 10:15:00 | 526.50 | 529.26 | 522.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-09 14:15:00 | 534.90 | 528.53 | 522.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 534.90 | 528.53 | 522.47 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-08-14 11:15:00 | 539.60 | 529.12 | 523.30 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 583.60 | 600.07 | 578.81 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-04 14:15:00 | 575.40 | 599.15 | 578.87 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 533.30 | 569.60 | 569.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 523.75 | 564.27 | 566.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 468.90 | 464.95 | 487.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-10 09:15:00 | 461.50 | 465.17 | 486.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 479.65 | 466.06 | 481.26 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-23 10:15:00 | 482.75 | 466.22 | 481.27 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 12:15:00 | 489.35 | 483.59 | 483.59 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-05 15:15:00 | 482.10 | 483.58 | 483.58 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 09:15:00 | 495.00 | 483.70 | 483.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 497.95 | 483.84 | 483.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 10:15:00 | 485.30 | 488.36 | 486.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-17 09:15:00 | 497.40 | 488.10 | 486.17 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 13:15:00 | 492.20 | 495.92 | 491.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-02 09:15:00 | 494.50 | 495.86 | 491.45 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-19 12:15:00 | 547.00 | 559.81 | 547.32 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 536.50 | 560.18 | 560.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 530.65 | 556.75 | 558.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 13:15:00 | 544.70 | 543.97 | 549.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-15 11:15:00 | 538.50 | 544.19 | 549.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 551.25 | 544.21 | 549.37 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 15:15:00 | 577.90 | 541.74 | 541.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 10:15:00 | 587.00 | 547.22 | 544.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 10:15:00 | 560.30 | 560.87 | 553.34 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 537.80 | 549.61 | 549.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 535.25 | 547.99 | 548.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 431.00 | 430.95 | 454.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-07 09:15:00 | 426.95 | 430.91 | 454.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 452.00 | 432.11 | 452.66 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 11:15:00 | 454.30 | 432.33 | 452.67 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-23 09:15:00 | 574.10 | 2023-10-26 10:15:00 | 554.24 | TARGET | 19.86 |
| SELL | 2023-10-23 14:15:00 | 569.70 | 2023-10-26 15:15:00 | 537.12 | TARGET | 32.58 |
| BUY | 2023-12-14 14:15:00 | 580.30 | 2023-12-18 12:15:00 | 592.89 | TARGET | 12.59 |
| BUY | 2023-12-14 15:15:00 | 581.00 | 2023-12-18 13:15:00 | 595.62 | TARGET | 14.62 |
| SELL | 2024-02-22 09:15:00 | 563.45 | 2024-02-22 14:15:00 | 571.55 | EXIT_EMA400 | -8.10 |
| BUY | 2024-08-09 14:15:00 | 534.90 | 2024-08-21 09:15:00 | 572.18 | TARGET | 37.28 |
| BUY | 2024-08-14 11:15:00 | 539.60 | 2024-08-22 09:15:00 | 588.51 | TARGET | 48.91 |
| SELL | 2025-01-10 09:15:00 | 461.50 | 2025-01-23 10:15:00 | 482.75 | EXIT_EMA400 | -21.25 |
| BUY | 2025-04-02 09:15:00 | 494.50 | 2025-04-02 11:15:00 | 503.65 | TARGET | 9.15 |
| BUY | 2025-03-17 09:15:00 | 497.40 | 2025-04-08 11:15:00 | 531.10 | TARGET | 33.70 |
| SELL | 2025-09-15 11:15:00 | 538.50 | 2025-09-16 09:15:00 | 551.25 | EXIT_EMA400 | -12.75 |
| SELL | 2026-04-07 09:15:00 | 426.95 | 2026-04-10 11:15:00 | 454.30 | EXIT_EMA400 | -27.35 |
