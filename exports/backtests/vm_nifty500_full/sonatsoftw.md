# Sonata Software Ltd. (SONATSOFTW.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 255.03
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 2 / 5
- **Target hits / EMA400 exits:** 2 / 5
- **Total realized P&L (per unit):** -72.55
- **Avg P&L per closed trade:** -10.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 716.00 | 753.41 | 753.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-16 14:15:00 | 701.10 | 748.32 | 750.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 573.00 | 563.10 | 617.67 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 15:15:00 | 700.00 | 619.14 | 618.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 709.05 | 632.28 | 625.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 648.95 | 663.68 | 644.70 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 629.80 | 634.83 | 634.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 626.80 | 634.69 | 634.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 10:15:00 | 645.00 | 634.79 | 634.83 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-28 15:15:00 | 624.50 | 634.59 | 634.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-29 09:15:00 | 648.55 | 634.73 | 634.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 10:15:00 | 652.80 | 634.91 | 634.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 11:15:00 | 655.05 | 635.11 | 634.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 11:15:00 | 654.95 | 659.38 | 649.93 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 601.25 | 644.17 | 644.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 587.60 | 643.60 | 644.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-16 14:15:00 | 627.30 | 624.75 | 633.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-17 15:15:00 | 617.95 | 624.67 | 632.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 617.05 | 612.83 | 623.68 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 09:15:00 | 599.70 | 612.64 | 623.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-11-06 13:15:00 | 623.50 | 611.98 | 621.95 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 670.10 | 616.90 | 616.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 673.05 | 617.97 | 617.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 634.00 | 640.72 | 630.68 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 609.00 | 624.36 | 624.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 596.85 | 623.65 | 624.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-24 11:15:00 | 368.20 | 349.24 | 397.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-25 09:15:00 | 340.20 | 349.50 | 396.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-30 10:15:00 | 397.10 | 351.82 | 392.55 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 13:15:00 | 416.00 | 401.69 | 401.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 13:15:00 | 424.25 | 403.34 | 402.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 404.85 | 406.05 | 404.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-25 10:15:00 | 414.05 | 404.36 | 403.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 404.80 | 406.42 | 404.65 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-04 10:15:00 | 410.15 | 406.40 | 404.76 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-25 12:15:00 | 414.80 | 422.39 | 415.47 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 356.75 | 410.34 | 410.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 354.00 | 408.24 | 409.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 384.20 | 380.61 | 391.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 09:15:00 | 374.70 | 380.84 | 391.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-12 09:15:00 | 385.20 | 371.11 | 381.77 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-28 15:15:00 | 624.50 | 2024-08-29 09:15:00 | 648.55 | EXIT_EMA400 | -24.05 |
| SELL | 2024-10-17 15:15:00 | 617.95 | 2024-11-06 13:15:00 | 623.50 | EXIT_EMA400 | -5.55 |
| SELL | 2024-11-04 09:15:00 | 599.70 | 2024-11-06 13:15:00 | 623.50 | EXIT_EMA400 | -23.80 |
| SELL | 2025-04-25 09:15:00 | 340.20 | 2025-04-30 10:15:00 | 397.10 | EXIT_EMA400 | -56.90 |
| BUY | 2025-07-04 10:15:00 | 410.15 | 2025-07-08 09:15:00 | 426.31 | TARGET | 16.16 |
| BUY | 2025-06-25 10:15:00 | 414.05 | 2025-07-08 12:15:00 | 446.13 | TARGET | 32.08 |
| SELL | 2025-08-26 09:15:00 | 374.70 | 2025-09-12 09:15:00 | 385.20 | EXIT_EMA400 | -10.50 |
