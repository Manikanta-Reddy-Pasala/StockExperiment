# Sona BLW Precision Forgings Ltd. (SONACOMS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 607.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 9 |
| ENTRY1 | 11 |
| ENTRY2 | 3 |
| EXIT | 11 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 1 / 13
- **Target hits / EMA400 exits:** 1 / 13
- **Total realized P&L (per unit):** -211.07
- **Avg P&L per closed trade:** -15.08

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 12:15:00 | 523.80 | 566.95 | 567.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 13:15:00 | 522.50 | 566.51 | 566.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 560.60 | 560.06 | 563.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-30 13:15:00 | 543.00 | 559.19 | 562.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 556.35 | 555.25 | 559.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-07 11:15:00 | 561.50 | 555.36 | 559.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-20 15:15:00 | 581.25 | 563.22 | 563.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-21 10:15:00 | 595.70 | 563.69 | 563.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-28 09:15:00 | 562.15 | 566.77 | 565.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-05 09:15:00 | 574.55 | 564.69 | 564.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 574.55 | 564.69 | 564.21 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2023-12-05 10:15:00 | 575.80 | 564.80 | 564.27 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2023-12-12 09:15:00 | 564.25 | 567.60 | 565.89 | Close below EMA400 |

### Cycle 3 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 538.30 | 564.31 | 564.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 537.00 | 564.04 | 564.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 10:15:00 | 568.75 | 564.02 | 564.26 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 578.50 | 564.54 | 564.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 11:15:00 | 585.50 | 565.61 | 565.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 09:15:00 | 598.45 | 604.48 | 589.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-31 12:15:00 | 615.95 | 596.42 | 589.16 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 602.20 | 607.56 | 596.91 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-09 10:15:00 | 595.15 | 607.43 | 596.90 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-10 09:15:00 | 596.70 | 644.10 | 644.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-10 11:15:00 | 590.25 | 643.06 | 643.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 09:15:00 | 628.00 | 627.11 | 634.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 11:15:00 | 608.90 | 631.49 | 634.93 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-06-04 12:15:00 | 638.35 | 631.56 | 634.94 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 14:15:00 | 660.35 | 637.97 | 637.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 665.65 | 638.47 | 638.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-18 13:15:00 | 640.60 | 642.50 | 640.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-19 09:15:00 | 646.40 | 642.52 | 640.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-20 09:15:00 | 637.00 | 642.54 | 640.53 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 13:15:00 | 645.95 | 694.32 | 694.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 640.60 | 692.89 | 693.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 09:15:00 | 708.15 | 680.26 | 686.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 12:15:00 | 664.45 | 689.20 | 690.01 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-19 09:15:00 | 693.90 | 685.82 | 688.17 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 13:15:00 | 527.70 | 501.62 | 501.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 528.75 | 501.89 | 501.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 512.30 | 521.23 | 513.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 12:15:00 | 521.30 | 520.70 | 513.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 515.85 | 520.66 | 513.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-10 13:15:00 | 518.45 | 520.48 | 513.51 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-12 12:15:00 | 513.00 | 520.54 | 513.98 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 486.95 | 508.82 | 508.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 480.30 | 508.54 | 508.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 11:15:00 | 487.95 | 477.57 | 489.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-23 09:15:00 | 470.30 | 478.48 | 488.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 487.70 | 478.44 | 488.13 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-23 13:15:00 | 489.10 | 478.55 | 488.13 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 483.80 | 449.57 | 449.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-07 12:15:00 | 487.35 | 460.99 | 455.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 11:15:00 | 489.30 | 490.68 | 477.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 09:15:00 | 496.25 | 490.81 | 478.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 480.50 | 490.56 | 478.45 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-12-09 11:15:00 | 487.05 | 490.46 | 478.52 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 479.30 | 490.24 | 478.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-10 10:15:00 | 476.45 | 489.89 | 478.59 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 10:15:00 | 460.65 | 477.74 | 477.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 452.40 | 475.45 | 476.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 15:15:00 | 470.00 | 469.69 | 473.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-23 09:15:00 | 463.95 | 469.64 | 473.33 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 470.80 | 469.06 | 472.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-27 10:15:00 | 477.45 | 469.14 | 472.94 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 531.90 | 476.45 | 476.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 535.25 | 479.56 | 477.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 513.15 | 517.17 | 503.05 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-11 09:15:00 | 527.10 | 514.37 | 504.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 506.05 | 514.57 | 504.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-03-13 09:15:00 | 501.65 | 514.21 | 504.68 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-30 13:15:00 | 543.00 | 2023-11-07 11:15:00 | 561.50 | EXIT_EMA400 | -18.50 |
| BUY | 2023-12-05 09:15:00 | 574.55 | 2023-12-12 09:15:00 | 564.25 | EXIT_EMA400 | -10.30 |
| BUY | 2023-12-05 10:15:00 | 575.80 | 2023-12-12 09:15:00 | 564.25 | EXIT_EMA400 | -11.55 |
| BUY | 2024-01-31 12:15:00 | 615.95 | 2024-02-09 10:15:00 | 595.15 | EXIT_EMA400 | -20.80 |
| SELL | 2024-06-04 11:15:00 | 608.90 | 2024-06-04 12:15:00 | 638.35 | EXIT_EMA400 | -29.45 |
| BUY | 2024-06-19 09:15:00 | 646.40 | 2024-06-20 09:15:00 | 637.00 | EXIT_EMA400 | -9.40 |
| SELL | 2024-11-13 12:15:00 | 664.45 | 2024-11-19 09:15:00 | 693.90 | EXIT_EMA400 | -29.45 |
| BUY | 2025-06-10 13:15:00 | 518.45 | 2025-06-11 09:15:00 | 533.28 | TARGET | 14.83 |
| BUY | 2025-06-09 12:15:00 | 521.30 | 2025-06-12 12:15:00 | 513.00 | EXIT_EMA400 | -8.30 |
| SELL | 2025-07-23 09:15:00 | 470.30 | 2025-07-23 13:15:00 | 489.10 | EXIT_EMA400 | -18.80 |
| BUY | 2025-12-08 09:15:00 | 496.25 | 2025-12-10 10:15:00 | 476.45 | EXIT_EMA400 | -19.80 |
| BUY | 2025-12-09 11:15:00 | 487.05 | 2025-12-10 10:15:00 | 476.45 | EXIT_EMA400 | -10.60 |
| SELL | 2026-01-23 09:15:00 | 463.95 | 2026-01-27 10:15:00 | 477.45 | EXIT_EMA400 | -13.50 |
| BUY | 2026-03-11 09:15:00 | 527.10 | 2026-03-13 09:15:00 | 501.65 | EXIT_EMA400 | -25.45 |
