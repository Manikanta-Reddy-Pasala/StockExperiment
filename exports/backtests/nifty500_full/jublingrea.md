# Jubilant Ingrevia Ltd. (JUBLINGREA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 713.75
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 13 |
| ENTRY1 | 11 |
| ENTRY2 | 7 |
| EXIT | 11 |

## P&L

- **Trades closed:** 18
- **Trades open at end:** 0
- **Winners / losers:** 7 / 11
- **Target hits / EMA400 exits:** 7 / 11
- **Total realized P&L (per unit):** 74.09
- **Avg P&L per closed trade:** 4.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 10:15:00 | 426.90 | 462.77 | 462.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 11:15:00 | 419.85 | 462.34 | 462.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 436.35 | 435.01 | 444.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-24 12:15:00 | 433.40 | 435.01 | 444.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 435.95 | 433.40 | 442.27 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-12-05 11:15:00 | 433.95 | 433.75 | 442.06 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 15:15:00 | 441.00 | 433.91 | 441.98 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-12-06 09:15:00 | 444.25 | 434.02 | 441.99 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-19 13:15:00 | 481.00 | 447.11 | 447.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-20 12:15:00 | 483.20 | 449.11 | 448.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-09 09:15:00 | 475.90 | 477.71 | 465.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-11 09:15:00 | 487.55 | 478.31 | 467.03 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-18 09:15:00 | 465.10 | 480.16 | 469.90 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 12:15:00 | 432.15 | 463.97 | 464.00 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-02-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 15:15:00 | 474.75 | 462.90 | 462.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 09:15:00 | 484.20 | 463.11 | 462.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 13:15:00 | 467.00 | 467.21 | 465.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-26 12:15:00 | 485.60 | 467.25 | 465.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 12:15:00 | 485.60 | 467.25 | 465.34 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-06 10:15:00 | 467.75 | 472.17 | 468.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-14 11:15:00 | 437.55 | 465.73 | 465.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-14 12:15:00 | 433.10 | 465.41 | 465.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 13:15:00 | 458.75 | 457.64 | 461.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-26 09:15:00 | 452.60 | 457.62 | 461.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 452.60 | 457.62 | 461.10 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-03-27 12:15:00 | 446.20 | 457.00 | 460.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 458.35 | 456.10 | 459.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-04-01 10:15:00 | 461.50 | 456.15 | 459.96 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 487.75 | 462.86 | 462.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 494.50 | 463.17 | 463.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 519.00 | 519.48 | 500.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-10 11:15:00 | 523.95 | 518.65 | 500.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 514.90 | 521.64 | 507.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-27 13:15:00 | 507.40 | 520.45 | 507.43 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 10:15:00 | 690.30 | 753.46 | 753.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 11:15:00 | 686.95 | 752.80 | 753.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 734.00 | 727.97 | 739.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 11:15:00 | 712.50 | 728.70 | 737.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 691.50 | 670.50 | 696.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-07 13:15:00 | 685.75 | 670.88 | 696.70 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-19 09:15:00 | 693.00 | 672.60 | 692.41 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 692.50 | 681.37 | 681.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 701.75 | 681.77 | 681.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 13:15:00 | 676.50 | 682.97 | 682.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 720.55 | 682.48 | 682.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 720.55 | 682.48 | 682.03 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-02 11:15:00 | 729.05 | 687.54 | 684.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-06-11 11:15:00 | 682.80 | 696.83 | 690.74 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 15:15:00 | 701.00 | 745.01 | 745.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 12:15:00 | 690.10 | 723.46 | 731.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 09:15:00 | 686.00 | 680.88 | 701.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-07 13:15:00 | 648.25 | 684.55 | 695.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-11-17 09:15:00 | 708.95 | 680.76 | 691.13 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 707.60 | 698.39 | 698.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 719.00 | 698.69 | 698.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 11:15:00 | 694.40 | 700.00 | 699.25 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 683.30 | 698.47 | 698.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 680.55 | 698.30 | 698.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 694.65 | 692.21 | 695.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 09:15:00 | 678.55 | 693.58 | 695.49 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 09:15:00 | 698.70 | 692.71 | 694.98 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 14:15:00 | 710.60 | 697.06 | 697.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 09:15:00 | 722.70 | 697.43 | 697.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 11:15:00 | 697.50 | 699.32 | 698.19 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-31 10:15:00 | 708.50 | 699.30 | 698.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 706.25 | 710.02 | 704.33 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-09 09:15:00 | 700.00 | 709.92 | 704.34 | Close below EMA400 |

### Cycle 13 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 670.00 | 699.87 | 699.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 667.45 | 699.55 | 699.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 674.05 | 671.37 | 683.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-04 09:15:00 | 655.50 | 670.91 | 683.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 665.50 | 663.31 | 677.49 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-10 11:15:00 | 663.95 | 663.35 | 677.37 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 675.85 | 663.43 | 676.99 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 09:15:00 | 663.15 | 663.73 | 676.75 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 669.25 | 663.84 | 676.61 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-12 14:15:00 | 664.00 | 663.93 | 676.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 606.05 | 584.21 | 608.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 619.95 | 584.56 | 608.33 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 735.05 | 624.44 | 624.19 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-24 12:15:00 | 433.40 | 2023-12-06 09:15:00 | 444.25 | EXIT_EMA400 | -10.85 |
| SELL | 2023-12-05 11:15:00 | 433.95 | 2023-12-06 09:15:00 | 444.25 | EXIT_EMA400 | -10.30 |
| BUY | 2024-01-11 09:15:00 | 487.55 | 2024-01-18 09:15:00 | 465.10 | EXIT_EMA400 | -22.45 |
| BUY | 2024-02-26 12:15:00 | 485.60 | 2024-03-06 10:15:00 | 467.75 | EXIT_EMA400 | -17.85 |
| SELL | 2024-03-26 09:15:00 | 452.60 | 2024-04-01 10:15:00 | 461.50 | EXIT_EMA400 | -8.90 |
| SELL | 2024-03-27 12:15:00 | 446.20 | 2024-04-01 10:15:00 | 461.50 | EXIT_EMA400 | -15.30 |
| BUY | 2024-05-10 11:15:00 | 523.95 | 2024-05-27 13:15:00 | 507.40 | EXIT_EMA400 | -16.55 |
| SELL | 2025-02-10 11:15:00 | 712.50 | 2025-02-14 09:15:00 | 636.70 | TARGET | 75.80 |
| SELL | 2025-03-07 13:15:00 | 685.75 | 2025-03-11 09:15:00 | 652.91 | TARGET | 32.84 |
| BUY | 2025-05-28 09:15:00 | 720.55 | 2025-06-11 11:15:00 | 682.80 | EXIT_EMA400 | -37.75 |
| BUY | 2025-06-02 11:15:00 | 729.05 | 2025-06-11 11:15:00 | 682.80 | EXIT_EMA400 | -46.25 |
| SELL | 2025-11-07 13:15:00 | 648.25 | 2025-11-17 09:15:00 | 708.95 | EXIT_EMA400 | -60.70 |
| SELL | 2025-12-18 09:15:00 | 678.55 | 2025-12-19 09:15:00 | 698.70 | EXIT_EMA400 | -20.15 |
| BUY | 2025-12-31 10:15:00 | 708.50 | 2026-01-05 09:15:00 | 739.36 | TARGET | 30.86 |
| SELL | 2026-02-12 14:15:00 | 664.00 | 2026-02-16 10:15:00 | 626.42 | TARGET | 37.58 |
| SELL | 2026-02-10 11:15:00 | 663.95 | 2026-02-16 11:15:00 | 623.70 | TARGET | 40.25 |
| SELL | 2026-02-12 09:15:00 | 663.15 | 2026-02-18 15:15:00 | 622.36 | TARGET | 40.79 |
| SELL | 2026-02-04 09:15:00 | 655.50 | 2026-03-02 09:15:00 | 572.48 | TARGET | 83.02 |
