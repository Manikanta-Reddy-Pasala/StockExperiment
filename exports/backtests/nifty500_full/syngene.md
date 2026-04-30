# Syngene International Ltd. (SYNGENE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 467.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -63.90
- **Avg P&L per closed trade:** -10.65

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 09:15:00 | 709.25 | 786.39 | 786.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 12:15:00 | 705.80 | 779.30 | 783.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 733.80 | 726.87 | 746.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-11 13:15:00 | 715.60 | 731.93 | 741.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-01-03 13:15:00 | 728.20 | 712.76 | 724.93 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-02-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 13:15:00 | 749.25 | 724.84 | 724.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 752.95 | 729.16 | 727.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 11:15:00 | 737.55 | 738.62 | 732.97 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 11:15:00 | 687.15 | 728.63 | 728.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 682.95 | 726.68 | 727.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 09:15:00 | 707.95 | 701.69 | 712.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-24 15:15:00 | 693.00 | 713.67 | 715.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-22 13:15:00 | 699.70 | 689.59 | 698.96 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 712.50 | 697.17 | 697.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 714.95 | 697.35 | 697.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 875.00 | 877.82 | 838.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-30 12:15:00 | 895.65 | 878.72 | 843.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-21 14:15:00 | 854.35 | 880.36 | 858.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 10:15:00 | 854.40 | 873.19 | 873.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 849.35 | 872.55 | 872.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 12:15:00 | 872.60 | 871.19 | 872.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-02 09:15:00 | 862.45 | 871.26 | 872.24 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 862.45 | 871.26 | 872.24 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 14:15:00 | 873.35 | 871.08 | 872.12 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 709.50 | 660.08 | 659.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 711.45 | 660.59 | 660.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 674.25 | 674.85 | 668.20 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 11:15:00 | 632.50 | 664.96 | 665.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 629.70 | 664.61 | 664.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 663.35 | 658.93 | 661.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 15:15:00 | 653.80 | 658.87 | 661.63 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 657.00 | 658.85 | 661.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-05 11:15:00 | 647.70 | 658.69 | 661.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 658.70 | 656.57 | 660.02 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-11 11:15:00 | 663.60 | 656.67 | 660.04 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 15:15:00 | 658.55 | 646.00 | 645.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 661.90 | 648.30 | 647.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 647.45 | 650.24 | 648.41 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 10:15:00 | 625.45 | 646.64 | 646.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 618.45 | 641.88 | 644.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 15:15:00 | 415.00 | 413.94 | 451.15 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-11 13:15:00 | 715.60 | 2024-01-03 13:15:00 | 728.20 | EXIT_EMA400 | -12.60 |
| SELL | 2024-04-24 15:15:00 | 693.00 | 2024-05-22 13:15:00 | 699.70 | EXIT_EMA400 | -6.70 |
| BUY | 2024-09-30 12:15:00 | 895.65 | 2024-10-21 14:15:00 | 854.35 | EXIT_EMA400 | -41.30 |
| SELL | 2025-01-02 09:15:00 | 862.45 | 2025-01-02 14:15:00 | 873.35 | EXIT_EMA400 | -10.90 |
| SELL | 2025-09-04 15:15:00 | 653.80 | 2025-09-09 11:15:00 | 630.30 | TARGET | 23.50 |
| SELL | 2025-09-05 11:15:00 | 647.70 | 2025-09-11 11:15:00 | 663.60 | EXIT_EMA400 | -15.90 |
