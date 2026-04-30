# Sun TV Network Ltd. (SUNTV.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 605.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 11 |
| ENTRY1 | 6 |
| ENTRY2 | 6 |
| EXIT | 6 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** 61.98
- **Avg P&L per closed trade:** 5.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-09 10:15:00 | 622.40 | 667.56 | 667.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 09:15:00 | 615.85 | 664.97 | 666.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-04 13:15:00 | 638.15 | 635.62 | 647.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-06 09:15:00 | 621.70 | 636.15 | 646.89 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-03 09:15:00 | 625.35 | 609.39 | 624.87 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 14:15:00 | 653.20 | 627.26 | 627.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-09 09:15:00 | 656.35 | 629.21 | 628.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 645.70 | 649.45 | 640.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-30 13:15:00 | 672.50 | 650.33 | 642.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 653.15 | 652.99 | 644.10 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-04 12:15:00 | 670.90 | 653.07 | 644.27 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 790.35 | 832.33 | 787.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-13 12:15:00 | 782.40 | 831.42 | 787.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 754.00 | 801.37 | 801.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 747.70 | 798.61 | 799.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 15:15:00 | 770.00 | 768.45 | 781.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-08 09:15:00 | 753.50 | 767.94 | 780.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 12:15:00 | 767.20 | 753.24 | 768.56 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-25 14:15:00 | 759.95 | 753.44 | 768.51 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 765.40 | 753.55 | 765.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-03 11:15:00 | 773.75 | 753.75 | 765.99 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 15:15:00 | 667.65 | 634.34 | 634.22 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 589.85 | 635.32 | 635.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 586.10 | 613.64 | 621.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 14:15:00 | 591.20 | 590.06 | 603.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-22 12:15:00 | 587.70 | 590.26 | 603.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 567.55 | 574.68 | 589.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 14:15:00 | 554.60 | 575.41 | 585.44 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 571.55 | 569.85 | 581.41 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-08 11:15:00 | 552.30 | 567.61 | 578.63 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 558.75 | 552.49 | 564.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-01 12:15:00 | 567.15 | 552.63 | 564.75 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 575.20 | 560.48 | 560.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 09:15:00 | 584.35 | 560.71 | 560.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 09:15:00 | 564.85 | 567.43 | 564.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 540.35 | 561.97 | 561.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 539.25 | 561.15 | 561.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 10:15:00 | 560.25 | 559.07 | 560.43 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-27 09:15:00 | 529.90 | 558.77 | 560.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 555.75 | 553.98 | 557.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-02 09:15:00 | 536.75 | 553.81 | 557.36 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 548.60 | 553.17 | 556.91 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 14:15:00 | 544.70 | 552.99 | 556.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 554.70 | 550.22 | 554.87 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-09 10:15:00 | 558.00 | 550.30 | 554.88 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 611.25 | 559.00 | 558.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 635.55 | 575.77 | 568.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 576.00 | 585.18 | 575.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-18 15:15:00 | 604.00 | 581.04 | 575.25 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 586.00 | 587.46 | 579.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-30 10:15:00 | 574.75 | 587.34 | 579.92 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-06 09:15:00 | 621.70 | 2024-04-03 09:15:00 | 625.35 | EXIT_EMA400 | -3.65 |
| BUY | 2024-06-04 12:15:00 | 670.90 | 2024-06-05 14:15:00 | 750.78 | TARGET | 79.88 |
| BUY | 2024-05-30 13:15:00 | 672.50 | 2024-06-12 09:15:00 | 763.87 | TARGET | 91.37 |
| SELL | 2024-11-08 09:15:00 | 753.50 | 2024-12-03 11:15:00 | 773.75 | EXIT_EMA400 | -20.25 |
| SELL | 2024-11-25 14:15:00 | 759.95 | 2024-12-03 11:15:00 | 773.75 | EXIT_EMA400 | -13.80 |
| SELL | 2025-07-22 12:15:00 | 587.70 | 2025-08-29 09:15:00 | 539.97 | TARGET | 47.73 |
| SELL | 2025-08-26 14:15:00 | 554.60 | 2025-10-01 12:15:00 | 567.15 | EXIT_EMA400 | -12.55 |
| SELL | 2025-09-08 11:15:00 | 552.30 | 2025-10-01 12:15:00 | 567.15 | EXIT_EMA400 | -14.85 |
| SELL | 2026-01-27 09:15:00 | 529.90 | 2026-02-09 10:15:00 | 558.00 | EXIT_EMA400 | -28.10 |
| SELL | 2026-02-02 09:15:00 | 536.75 | 2026-02-09 10:15:00 | 558.00 | EXIT_EMA400 | -21.25 |
| SELL | 2026-02-03 14:15:00 | 544.70 | 2026-02-09 10:15:00 | 558.00 | EXIT_EMA400 | -13.30 |
| BUY | 2026-03-18 15:15:00 | 604.00 | 2026-03-30 10:15:00 | 574.75 | EXIT_EMA400 | -29.25 |
