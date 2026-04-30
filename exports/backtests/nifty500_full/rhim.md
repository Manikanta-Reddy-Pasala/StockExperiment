# RHI MAGNESITA INDIA LTD. (RHIM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 403.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 13 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 8 |
| ENTRY1 | 9 |
| ENTRY2 | 5 |
| EXIT | 9 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 1 / 13
- **Target hits / EMA400 exits:** 1 / 13
- **Total realized P&L (per unit):** -204.09
- **Avg P&L per closed trade:** -14.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-15 14:15:00 | 689.95 | 705.52 | 705.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-16 09:15:00 | 685.70 | 705.15 | 705.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 10:15:00 | 702.85 | 701.80 | 703.59 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-11-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 12:15:00 | 740.40 | 705.52 | 705.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 11:15:00 | 748.00 | 711.36 | 708.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 10:15:00 | 772.55 | 773.48 | 753.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 11:15:00 | 777.30 | 773.51 | 753.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 13:15:00 | 760.45 | 774.24 | 756.00 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-01-17 12:15:00 | 773.85 | 773.67 | 756.25 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 754.00 | 773.24 | 756.38 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 09:15:00 | 699.95 | 746.87 | 746.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 13:15:00 | 692.65 | 744.98 | 745.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 11:15:00 | 585.00 | 580.28 | 624.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 13:15:00 | 578.50 | 580.27 | 623.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 14:15:00 | 617.25 | 588.73 | 620.06 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-12 10:15:00 | 620.05 | 589.61 | 620.04 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 668.00 | 630.94 | 630.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 10:15:00 | 675.50 | 634.22 | 632.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-05 14:15:00 | 655.65 | 657.29 | 646.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 673.65 | 657.44 | 646.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-06-24 12:15:00 | 654.90 | 666.53 | 655.58 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 612.50 | 648.81 | 648.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 13:15:00 | 610.60 | 648.43 | 648.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 11:15:00 | 619.60 | 617.82 | 629.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-29 12:15:00 | 614.45 | 617.79 | 629.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-16 09:15:00 | 641.00 | 604.57 | 617.56 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 478.15 | 462.70 | 462.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 484.80 | 464.38 | 463.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 09:15:00 | 465.45 | 465.64 | 464.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 444.50 | 462.89 | 462.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 14:15:00 | 441.60 | 461.42 | 462.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 12:15:00 | 457.15 | 455.25 | 458.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-13 12:15:00 | 449.30 | 454.98 | 458.35 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-14 09:15:00 | 464.15 | 454.91 | 458.25 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 468.45 | 460.59 | 460.55 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 14:15:00 | 458.20 | 460.50 | 460.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 14:15:00 | 445.65 | 460.12 | 460.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 456.70 | 456.68 | 458.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-09 13:15:00 | 448.25 | 456.48 | 458.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 456.60 | 456.27 | 458.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-10 13:15:00 | 452.95 | 456.21 | 458.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 456.90 | 456.18 | 458.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-11 11:15:00 | 452.60 | 456.15 | 458.00 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-11 14:15:00 | 460.80 | 456.16 | 457.98 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-06-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 13:15:00 | 510.00 | 459.87 | 459.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 521.80 | 461.79 | 460.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 10:15:00 | 476.00 | 476.31 | 469.90 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-03 13:15:00 | 482.50 | 475.23 | 470.04 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 471.55 | 475.39 | 470.51 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-07 15:15:00 | 466.80 | 475.30 | 470.49 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 470.75 | 484.51 | 484.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 467.50 | 481.59 | 482.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 493.00 | 466.96 | 474.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-09 09:15:00 | 447.70 | 467.84 | 474.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 460.60 | 457.46 | 465.70 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-29 13:15:00 | 465.70 | 457.70 | 465.66 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 483.65 | 469.76 | 469.76 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 454.35 | 469.75 | 469.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 453.00 | 469.42 | 469.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 10:15:00 | 469.00 | 468.63 | 469.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 09:15:00 | 462.00 | 468.18 | 468.90 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 454.80 | 453.67 | 459.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-22 11:15:00 | 451.15 | 453.64 | 459.77 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 455.50 | 453.52 | 459.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-24 10:15:00 | 450.40 | 453.44 | 459.28 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-31 10:15:00 | 459.60 | 452.21 | 457.86 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-11 11:15:00 | 777.30 | 2024-01-18 09:15:00 | 754.00 | EXIT_EMA400 | -23.30 |
| BUY | 2024-01-17 12:15:00 | 773.85 | 2024-01-18 09:15:00 | 754.00 | EXIT_EMA400 | -19.85 |
| SELL | 2024-04-02 13:15:00 | 578.50 | 2024-04-12 10:15:00 | 620.05 | EXIT_EMA400 | -41.55 |
| BUY | 2024-06-06 09:15:00 | 673.65 | 2024-06-24 12:15:00 | 654.90 | EXIT_EMA400 | -18.75 |
| SELL | 2024-07-29 12:15:00 | 614.45 | 2024-08-16 09:15:00 | 641.00 | EXIT_EMA400 | -26.55 |
| SELL | 2025-05-13 12:15:00 | 449.30 | 2025-05-14 09:15:00 | 464.15 | EXIT_EMA400 | -14.85 |
| SELL | 2025-06-09 13:15:00 | 448.25 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -12.55 |
| SELL | 2025-06-10 13:15:00 | 452.95 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -7.85 |
| SELL | 2025-06-11 11:15:00 | 452.60 | 2025-06-11 14:15:00 | 460.80 | EXIT_EMA400 | -8.20 |
| BUY | 2025-07-03 13:15:00 | 482.50 | 2025-07-07 15:15:00 | 466.80 | EXIT_EMA400 | -15.70 |
| SELL | 2025-10-09 09:15:00 | 447.70 | 2025-10-29 13:15:00 | 465.70 | EXIT_EMA400 | -18.00 |
| SELL | 2025-12-02 09:15:00 | 462.00 | 2025-12-05 11:15:00 | 441.29 | TARGET | 20.71 |
| SELL | 2025-12-22 11:15:00 | 451.15 | 2025-12-31 10:15:00 | 459.60 | EXIT_EMA400 | -8.45 |
| SELL | 2025-12-24 10:15:00 | 450.40 | 2025-12-31 10:15:00 | 459.60 | EXIT_EMA400 | -9.20 |
