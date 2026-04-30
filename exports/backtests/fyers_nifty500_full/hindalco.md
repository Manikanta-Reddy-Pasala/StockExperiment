# Hindalco Industries Ltd. (HINDALCO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1037.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -14.57
- **Avg P&L per closed trade:** -2.43

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 09:15:00 | 625.65 | 665.56 | 665.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 10:15:00 | 616.10 | 662.18 | 663.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 09:15:00 | 653.80 | 650.15 | 656.92 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 710.10 | 662.16 | 662.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 713.45 | 676.39 | 671.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 14:15:00 | 719.10 | 722.01 | 704.15 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-23 09:15:00 | 730.00 | 722.07 | 704.36 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-24 09:15:00 | 680.85 | 721.63 | 704.75 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 11:15:00 | 659.40 | 694.73 | 694.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 12:15:00 | 655.00 | 694.33 | 694.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 13:15:00 | 672.10 | 669.43 | 678.38 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 660.55 | 669.49 | 677.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 677.15 | 669.27 | 677.28 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 10:15:00 | 666.60 | 669.43 | 677.05 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 599.25 | 600.35 | 615.40 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-14 11:15:00 | 597.70 | 600.33 | 615.25 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-19 09:15:00 | 615.80 | 601.27 | 614.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 15:15:00 | 680.45 | 622.89 | 622.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 691.25 | 623.57 | 623.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 10:15:00 | 668.05 | 668.42 | 651.53 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-15 10:15:00 | 610.15 | 639.84 | 639.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 09:15:00 | 610.00 | 638.36 | 639.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-30 10:15:00 | 630.95 | 630.94 | 634.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-30 11:15:00 | 627.20 | 630.91 | 634.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-02 09:15:00 | 647.65 | 630.80 | 634.34 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 657.50 | 636.11 | 636.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 12:15:00 | 659.50 | 637.15 | 636.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 09:15:00 | 637.20 | 645.55 | 641.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 651.95 | 642.78 | 640.81 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 645.10 | 645.75 | 642.66 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-13 10:15:00 | 641.45 | 645.70 | 642.65 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-23 09:15:00 | 730.00 | 2024-10-24 09:15:00 | 680.85 | EXIT_EMA400 | -49.15 |
| SELL | 2024-12-12 10:15:00 | 666.60 | 2024-12-18 09:15:00 | 635.24 | TARGET | 31.36 |
| SELL | 2024-12-09 09:15:00 | 660.55 | 2024-12-30 12:15:00 | 608.27 | TARGET | 52.28 |
| SELL | 2025-02-14 11:15:00 | 597.70 | 2025-02-19 09:15:00 | 615.80 | EXIT_EMA400 | -18.10 |
| SELL | 2025-04-30 11:15:00 | 627.20 | 2025-05-02 09:15:00 | 647.65 | EXIT_EMA400 | -20.45 |
| BUY | 2025-06-09 09:15:00 | 651.95 | 2025-06-13 10:15:00 | 641.45 | EXIT_EMA400 | -10.50 |
