# Five-Star Business Finance Ltd. (FIVESTAR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (4998 bars)
- **Last close:** 479.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 20 |
| ALERT1 | 16 |
| ALERT2 | 16 |
| ALERT3 | 8 |
| ENTRY1 | 10 |
| ENTRY2 | 4 |
| EXIT | 10 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 0 / 14
- **Target hits / EMA400 exits:** 0 / 14
- **Total realized P&L (per unit):** -312.95
- **Avg P&L per closed trade:** -22.35

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 15:15:00 | 696.00 | 743.00 | 743.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 10:15:00 | 688.25 | 741.99 | 742.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 10:15:00 | 732.85 | 731.36 | 736.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-02 10:15:00 | 723.05 | 731.53 | 736.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 13:15:00 | 735.30 | 731.61 | 736.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-01-02 14:15:00 | 724.00 | 731.54 | 736.19 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 13:15:00 | 731.50 | 730.56 | 735.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-01-08 09:15:00 | 730.65 | 730.62 | 735.19 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 729.75 | 729.87 | 734.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-11 13:15:00 | 739.70 | 730.01 | 734.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 11:15:00 | 754.30 | 737.23 | 737.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 15:15:00 | 756.80 | 737.88 | 737.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-12 14:15:00 | 749.85 | 754.07 | 747.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-02-13 15:15:00 | 764.20 | 753.86 | 747.18 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 09:15:00 | 751.50 | 753.84 | 747.20 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-14 10:15:00 | 741.05 | 753.71 | 747.17 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 11:15:00 | 709.50 | 743.41 | 743.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 15:15:00 | 708.50 | 740.17 | 741.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 10:15:00 | 687.85 | 687.57 | 709.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-19 09:15:00 | 680.60 | 713.61 | 716.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 712.85 | 707.54 | 712.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-26 12:15:00 | 709.25 | 707.56 | 712.93 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-26 14:15:00 | 733.85 | 707.85 | 713.02 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-03 13:15:00 | 773.60 | 717.57 | 717.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 09:15:00 | 782.85 | 719.34 | 718.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 14:15:00 | 719.10 | 728.39 | 723.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-13 09:15:00 | 742.40 | 728.34 | 723.58 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-13 11:15:00 | 723.00 | 728.35 | 723.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 14:15:00 | 720.20 | 765.46 | 765.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 716.30 | 761.45 | 763.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 11:15:00 | 738.85 | 738.44 | 749.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-29 11:15:00 | 732.80 | 738.34 | 747.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 739.30 | 738.36 | 747.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-30 12:15:00 | 753.75 | 738.53 | 747.57 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 798.20 | 751.97 | 751.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 15:15:00 | 815.00 | 753.07 | 752.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 09:15:00 | 781.30 | 833.81 | 806.23 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 648.75 | 784.39 | 784.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 645.25 | 772.73 | 778.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 670.00 | 668.18 | 703.31 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 802.00 | 724.46 | 724.41 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 14:15:00 | 652.75 | 725.35 | 725.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 644.65 | 723.93 | 724.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-24 14:15:00 | 699.65 | 696.59 | 708.72 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 15:15:00 | 760.75 | 716.98 | 716.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 09:15:00 | 763.85 | 717.44 | 717.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 695.45 | 720.98 | 719.04 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 707.05 | 717.21 | 717.21 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 722.85 | 717.24 | 717.23 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 12:15:00 | 700.00 | 717.10 | 717.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 685.40 | 716.24 | 716.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 709.40 | 707.13 | 711.68 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-04 10:15:00 | 700.00 | 713.05 | 714.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-05 09:15:00 | 715.95 | 712.33 | 713.76 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 754.15 | 706.13 | 706.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 12:15:00 | 759.40 | 707.12 | 706.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 09:15:00 | 731.50 | 743.61 | 727.70 | EMA200 retest candle locked |

### Cycle 15 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 679.90 | 716.46 | 716.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 10:15:00 | 675.50 | 715.00 | 715.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 703.60 | 701.95 | 708.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-28 11:15:00 | 695.70 | 702.79 | 708.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-30 14:15:00 | 708.60 | 700.65 | 706.65 | Close above EMA400 |

### Cycle 16 — BUY (started 2025-06-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 13:15:00 | 798.10 | 710.94 | 710.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 14:15:00 | 800.05 | 711.83 | 711.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 723.85 | 733.78 | 724.08 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 14:15:00 | 748.15 | 733.64 | 724.65 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 746.25 | 753.46 | 740.74 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-15 15:15:00 | 751.05 | 753.44 | 740.80 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 744.45 | 752.54 | 741.13 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-18 09:15:00 | 732.00 | 752.24 | 741.09 | Close below EMA400 |

### Cycle 17 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 650.05 | 733.41 | 733.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 13:15:00 | 644.15 | 732.52 | 732.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 12:15:00 | 540.30 | 536.68 | 567.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-20 13:15:00 | 533.85 | 536.65 | 566.88 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-29 13:15:00 | 584.05 | 536.53 | 561.69 | Close above EMA400 |

### Cycle 18 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 625.95 | 581.09 | 580.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 628.45 | 584.23 | 582.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 12:15:00 | 596.80 | 597.25 | 590.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-21 11:15:00 | 607.55 | 597.23 | 590.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-27 10:15:00 | 591.60 | 600.21 | 592.80 | Close below EMA400 |

### Cycle 19 — SELL (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 14:15:00 | 567.65 | 588.19 | 588.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-22 12:15:00 | 553.50 | 584.67 | 586.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 404.00 | 386.57 | 424.59 | EMA200 retest candle locked |

### Cycle 20 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 498.15 | 443.33 | 443.31 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-02 10:15:00 | 723.05 | 2024-01-11 13:15:00 | 739.70 | EXIT_EMA400 | -16.65 |
| SELL | 2024-01-02 14:15:00 | 724.00 | 2024-01-11 13:15:00 | 739.70 | EXIT_EMA400 | -15.70 |
| SELL | 2024-01-08 09:15:00 | 730.65 | 2024-01-11 13:15:00 | 739.70 | EXIT_EMA400 | -9.05 |
| BUY | 2024-02-13 15:15:00 | 764.20 | 2024-02-14 10:15:00 | 741.05 | EXIT_EMA400 | -23.15 |
| SELL | 2024-04-19 09:15:00 | 680.60 | 2024-04-26 14:15:00 | 733.85 | EXIT_EMA400 | -53.25 |
| SELL | 2024-04-26 12:15:00 | 709.25 | 2024-04-26 14:15:00 | 733.85 | EXIT_EMA400 | -24.60 |
| BUY | 2024-05-13 09:15:00 | 742.40 | 2024-05-13 11:15:00 | 723.00 | EXIT_EMA400 | -19.40 |
| SELL | 2024-08-29 11:15:00 | 732.80 | 2024-08-30 12:15:00 | 753.75 | EXIT_EMA400 | -20.95 |
| SELL | 2025-03-04 10:15:00 | 700.00 | 2025-03-05 09:15:00 | 715.95 | EXIT_EMA400 | -15.95 |
| SELL | 2025-05-28 11:15:00 | 695.70 | 2025-05-30 14:15:00 | 708.60 | EXIT_EMA400 | -12.90 |
| BUY | 2025-06-24 14:15:00 | 748.15 | 2025-07-18 09:15:00 | 732.00 | EXIT_EMA400 | -16.15 |
| BUY | 2025-07-15 15:15:00 | 751.05 | 2025-07-18 09:15:00 | 732.00 | EXIT_EMA400 | -19.05 |
| SELL | 2025-10-20 13:15:00 | 533.85 | 2025-10-29 13:15:00 | 584.05 | EXIT_EMA400 | -50.20 |
| BUY | 2025-11-21 11:15:00 | 607.55 | 2025-11-27 10:15:00 | 591.60 | EXIT_EMA400 | -15.95 |
