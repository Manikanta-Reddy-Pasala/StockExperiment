# Hindalco Industries Ltd. (HINDALCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1038.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 57.92
- **Avg P&L per closed trade:** 8.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 13:15:00 | 505.70 | 544.71 | 544.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 14:15:00 | 505.15 | 544.32 | 544.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 10:15:00 | 533.40 | 533.24 | 538.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-12 11:15:00 | 528.20 | 533.53 | 537.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 09:15:00 | 532.60 | 530.83 | 535.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-19 13:15:00 | 536.05 | 530.99 | 535.75 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 14:15:00 | 568.70 | 539.01 | 539.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-02 14:15:00 | 570.85 | 540.92 | 539.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 637.10 | 655.58 | 624.35 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-05 14:15:00 | 697.05 | 655.65 | 626.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-19 11:15:00 | 664.85 | 685.75 | 666.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 13:15:00 | 625.10 | 657.77 | 657.81 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 10:15:00 | 692.50 | 656.61 | 656.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 09:15:00 | 694.30 | 658.33 | 657.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 671.70 | 672.76 | 665.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-12 14:15:00 | 677.10 | 668.18 | 664.84 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-24 09:15:00 | 680.95 | 721.67 | 704.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 654.10 | 694.16 | 694.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 651.50 | 693.74 | 693.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 672.10 | 669.52 | 678.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 660.55 | 669.57 | 677.80 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 677.00 | 669.33 | 677.11 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 10:15:00 | 666.60 | 669.50 | 676.89 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-19 09:15:00 | 615.65 | 601.70 | 615.11 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-07 09:15:00 | 691.25 | 623.74 | 623.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 14:15:00 | 691.90 | 626.81 | 625.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.00 | 668.49 | 651.82 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 09:15:00 | 612.20 | 640.19 | 640.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 10:15:00 | 606.50 | 638.08 | 639.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 647.70 | 630.88 | 634.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-02 11:15:00 | 627.55 | 630.88 | 634.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-05 09:15:00 | 638.10 | 630.96 | 634.44 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 656.90 | 636.36 | 636.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.20 | 636.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.73 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 651.95 | 642.78 | 640.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 645.10 | 645.76 | 642.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 10:15:00 | 641.05 | 645.71 | 642.71 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-03-12 11:15:00 | 528.20 | 2024-03-19 13:15:00 | 536.05 | EXIT_EMA400 | -7.85 |
| BUY | 2024-06-05 14:15:00 | 697.05 | 2024-07-19 11:15:00 | 664.85 | EXIT_EMA400 | -32.20 |
| BUY | 2024-09-12 14:15:00 | 677.10 | 2024-09-24 13:15:00 | 713.89 | TARGET | 36.79 |
| SELL | 2024-12-12 10:15:00 | 666.60 | 2024-12-18 09:15:00 | 635.72 | TARGET | 30.88 |
| SELL | 2024-12-09 09:15:00 | 660.55 | 2024-12-30 12:15:00 | 608.80 | TARGET | 51.75 |
| SELL | 2025-05-02 11:15:00 | 627.55 | 2025-05-05 09:15:00 | 638.10 | EXIT_EMA400 | -10.55 |
| BUY | 2025-06-09 09:15:00 | 651.95 | 2025-06-13 10:15:00 | 641.05 | EXIT_EMA400 | -10.90 |
