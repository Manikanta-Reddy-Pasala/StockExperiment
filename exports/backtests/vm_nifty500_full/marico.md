# Marico Ltd. (MARICO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 775.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 2 |
| EXIT | 7 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -141.05
- **Avg P&L per closed trade:** -15.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 539.50 | 560.62 | 560.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 15:15:00 | 535.65 | 554.06 | 556.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 13:15:00 | 535.00 | 532.18 | 540.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-12-07 09:15:00 | 529.30 | 533.79 | 539.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 13:15:00 | 537.60 | 533.63 | 539.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-11 15:15:00 | 543.00 | 533.78 | 539.30 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-07 15:15:00 | 582.10 | 516.34 | 516.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 09:15:00 | 590.40 | 517.07 | 516.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 609.95 | 610.63 | 588.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-08 09:15:00 | 648.90 | 610.54 | 591.56 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-06 14:15:00 | 626.80 | 652.98 | 629.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 14:15:00 | 640.10 | 665.39 | 665.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 633.25 | 664.82 | 665.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 631.50 | 629.02 | 643.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 605.80 | 633.22 | 642.22 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-12 09:15:00 | 641.00 | 630.39 | 639.79 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 14:15:00 | 674.25 | 641.95 | 641.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 10:15:00 | 677.40 | 655.59 | 650.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 658.30 | 660.76 | 653.96 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 626.10 | 649.10 | 649.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 623.10 | 646.82 | 647.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 13:15:00 | 620.75 | 620.29 | 630.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-19 10:15:00 | 618.75 | 620.29 | 630.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-20 11:15:00 | 630.55 | 620.43 | 630.28 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 12:15:00 | 660.80 | 636.03 | 635.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-07 13:15:00 | 662.55 | 636.29 | 636.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 705.95 | 706.06 | 685.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-21 09:15:00 | 716.60 | 706.13 | 685.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 10:15:00 | 693.40 | 705.81 | 694.54 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 10:15:00 | 693.70 | 716.47 | 716.57 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 731.10 | 716.13 | 716.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 735.10 | 717.02 | 716.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 13:15:00 | 717.80 | 718.54 | 717.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-29 10:15:00 | 723.00 | 718.62 | 717.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 719.55 | 718.78 | 717.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-10-31 09:15:00 | 724.50 | 718.96 | 717.74 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 718.90 | 719.17 | 717.90 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-11-04 10:15:00 | 717.75 | 719.26 | 717.98 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 749.70 | 755.76 | 755.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-08 11:15:00 | 748.00 | 755.52 | 755.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 756.35 | 754.93 | 755.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-16 09:15:00 | 748.60 | 755.36 | 755.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 748.60 | 755.36 | 755.53 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-16 12:15:00 | 741.95 | 755.11 | 755.40 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-04-17 10:15:00 | 756.80 | 754.79 | 755.24 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 774.85 | 755.73 | 755.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 775.15 | 755.92 | 755.78 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-12-07 09:15:00 | 529.30 | 2023-12-11 15:15:00 | 543.00 | EXIT_EMA400 | -13.70 |
| BUY | 2024-07-08 09:15:00 | 648.90 | 2024-08-06 14:15:00 | 626.80 | EXIT_EMA400 | -22.10 |
| SELL | 2024-12-09 09:15:00 | 605.80 | 2024-12-12 09:15:00 | 641.00 | EXIT_EMA400 | -35.20 |
| SELL | 2025-03-19 10:15:00 | 618.75 | 2025-03-20 11:15:00 | 630.55 | EXIT_EMA400 | -11.80 |
| BUY | 2025-05-21 09:15:00 | 716.60 | 2025-06-12 10:15:00 | 693.40 | EXIT_EMA400 | -23.20 |
| BUY | 2025-10-29 10:15:00 | 723.00 | 2025-11-04 10:15:00 | 717.75 | EXIT_EMA400 | -5.25 |
| BUY | 2025-10-31 09:15:00 | 724.50 | 2025-11-04 10:15:00 | 717.75 | EXIT_EMA400 | -6.75 |
| SELL | 2026-04-16 09:15:00 | 748.60 | 2026-04-17 10:15:00 | 756.80 | EXIT_EMA400 | -8.20 |
| SELL | 2026-04-16 12:15:00 | 741.95 | 2026-04-17 10:15:00 | 756.80 | EXIT_EMA400 | -14.85 |
