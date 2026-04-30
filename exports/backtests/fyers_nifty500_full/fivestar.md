# Five-Star Business Finance Ltd. (FIVESTAR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 477.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 13 |
| ALERT2 | 12 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -91.99
- **Avg P&L per closed trade:** -13.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 11:15:00 | 724.70 | 770.14 | 770.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 12:15:00 | 723.35 | 769.67 | 770.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 738.85 | 738.53 | 750.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 11:15:00 | 732.80 | 738.43 | 749.25 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-08-30 12:15:00 | 753.75 | 738.61 | 748.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 806.65 | 753.55 | 753.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 11:15:00 | 819.00 | 759.35 | 756.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 781.10 | 833.80 | 806.52 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 652.50 | 784.18 | 784.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 648.75 | 782.83 | 784.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 670.00 | 667.87 | 703.08 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 802.00 | 724.34 | 724.26 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 652.75 | 725.29 | 725.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 644.65 | 723.89 | 724.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 699.65 | 696.53 | 708.64 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 756.00 | 717.07 | 716.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 14:15:00 | 762.55 | 718.35 | 717.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 13:15:00 | 721.45 | 722.65 | 719.87 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 685.00 | 717.36 | 717.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 667.00 | 708.33 | 711.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 704.00 | 693.47 | 702.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 09:15:00 | 687.65 | 700.74 | 704.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-08 10:15:00 | 722.45 | 697.66 | 702.54 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 11:15:00 | 752.70 | 706.73 | 706.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 12:15:00 | 759.40 | 707.25 | 706.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 731.10 | 743.69 | 727.87 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 677.40 | 716.86 | 716.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 675.50 | 715.03 | 715.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 703.60 | 701.97 | 708.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-28 14:15:00 | 688.15 | 702.51 | 708.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-30 14:15:00 | 708.60 | 700.65 | 706.72 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 798.10 | 710.94 | 710.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 800.05 | 711.83 | 711.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 723.85 | 733.71 | 724.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 14:15:00 | 748.15 | 733.58 | 724.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 746.25 | 753.34 | 740.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 15:15:00 | 750.50 | 753.31 | 740.73 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 744.45 | 752.43 | 741.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 732.00 | 752.12 | 741.03 | Close below EMA400 |

### Cycle 11 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 650.05 | 733.35 | 733.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 13:15:00 | 645.15 | 732.47 | 732.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 540.30 | 536.65 | 567.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-20 13:15:00 | 533.85 | 536.62 | 566.85 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 13:15:00 | 584.10 | 536.46 | 561.64 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 625.95 | 581.08 | 580.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 628.45 | 584.22 | 582.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 596.80 | 597.25 | 590.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-21 11:15:00 | 607.70 | 597.24 | 590.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-27 10:15:00 | 591.75 | 600.17 | 592.78 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 567.70 | 588.18 | 588.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 553.50 | 584.68 | 586.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 404.20 | 386.12 | 423.22 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 495.00 | 442.04 | 441.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 15:15:00 | 502.00 | 444.70 | 443.18 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-29 11:15:00 | 732.80 | 2024-08-30 12:15:00 | 753.75 | EXIT_EMA400 | -20.95 |
| SELL | 2025-04-04 09:15:00 | 687.65 | 2025-04-07 09:15:00 | 637.39 | TARGET | 50.26 |
| SELL | 2025-05-28 14:15:00 | 688.15 | 2025-05-30 14:15:00 | 708.60 | EXIT_EMA400 | -20.45 |
| BUY | 2025-06-24 14:15:00 | 748.15 | 2025-07-18 09:15:00 | 732.00 | EXIT_EMA400 | -16.15 |
| BUY | 2025-07-15 15:15:00 | 750.50 | 2025-07-18 09:15:00 | 732.00 | EXIT_EMA400 | -18.50 |
| SELL | 2025-10-20 13:15:00 | 533.85 | 2025-10-29 13:15:00 | 584.10 | EXIT_EMA400 | -50.25 |
| BUY | 2025-11-21 11:15:00 | 607.70 | 2025-11-27 10:15:00 | 591.75 | EXIT_EMA400 | -15.95 |
