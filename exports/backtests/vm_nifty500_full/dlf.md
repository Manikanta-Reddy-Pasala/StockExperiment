# DLF Ltd. (DLF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 587.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 7 |
| ENTRY1 | 10 |
| ENTRY2 | 5 |
| EXIT | 10 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 9 / 6
- **Target hits / EMA400 exits:** 8 / 7
- **Total realized P&L (per unit):** 127.12
- **Avg P&L per closed trade:** 8.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 13:15:00 | 484.00 | 486.91 | 486.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-23 14:15:00 | 480.80 | 486.85 | 486.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-24 10:15:00 | 489.75 | 486.79 | 486.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-08-24 12:15:00 | 484.70 | 486.77 | 486.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 484.70 | 486.77 | 486.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-08-24 14:15:00 | 483.65 | 486.73 | 486.82 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-08-29 09:15:00 | 493.60 | 485.87 | 486.36 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 500.95 | 486.92 | 486.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 13:15:00 | 502.25 | 487.07 | 486.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 12:15:00 | 538.10 | 541.06 | 525.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-30 11:15:00 | 543.80 | 538.46 | 526.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-15 11:15:00 | 813.95 | 863.00 | 814.46 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 13:15:00 | 805.70 | 851.70 | 851.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 802.05 | 851.20 | 851.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 11:15:00 | 850.00 | 848.20 | 849.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-04 09:15:00 | 839.70 | 848.63 | 850.13 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 839.70 | 848.63 | 850.13 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-06-04 10:15:00 | 784.50 | 847.99 | 849.80 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-06-07 15:15:00 | 845.00 | 838.75 | 844.52 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 12:15:00 | 856.95 | 848.88 | 848.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 15:15:00 | 861.00 | 849.20 | 849.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 841.25 | 851.44 | 850.22 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 825.15 | 849.00 | 849.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 816.75 | 846.18 | 847.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 09:15:00 | 842.60 | 841.58 | 844.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-04 10:15:00 | 837.20 | 841.53 | 844.83 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 839.35 | 840.08 | 843.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-07-10 10:15:00 | 830.90 | 839.99 | 843.57 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-07-16 10:15:00 | 848.85 | 838.10 | 842.10 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 12:15:00 | 873.80 | 842.98 | 842.98 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 813.10 | 843.00 | 843.02 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 15:15:00 | 867.00 | 842.32 | 842.27 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-09-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 15:15:00 | 814.20 | 842.97 | 843.04 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 10:15:00 | 860.50 | 842.75 | 842.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 13:15:00 | 864.95 | 843.28 | 842.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 11:15:00 | 842.55 | 844.73 | 843.75 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-20 09:15:00 | 858.00 | 844.85 | 843.84 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 868.60 | 871.73 | 859.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-04 09:15:00 | 845.00 | 871.31 | 859.53 | Close below EMA400 |

### Cycle 11 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 776.05 | 853.44 | 853.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 771.85 | 852.63 | 853.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 806.90 | 806.31 | 824.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-22 14:15:00 | 802.80 | 806.27 | 824.17 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 832.80 | 806.51 | 824.11 | Close above EMA400 |

### Cycle 12 — BUY (started 2024-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 13:15:00 | 868.85 | 832.86 | 832.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 871.05 | 835.27 | 834.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 828.05 | 845.84 | 840.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 09:15:00 | 851.80 | 845.77 | 840.11 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 851.80 | 845.77 | 840.11 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-23 12:15:00 | 838.70 | 845.70 | 840.15 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 10:15:00 | 810.25 | 836.67 | 836.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 11:15:00 | 805.05 | 836.35 | 836.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 769.10 | 764.05 | 788.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 14:15:00 | 764.15 | 764.19 | 787.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 706.85 | 684.47 | 715.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-21 14:15:00 | 695.65 | 685.23 | 715.06 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 685.25 | 667.62 | 690.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-24 09:15:00 | 683.05 | 669.25 | 690.16 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-02 09:15:00 | 687.20 | 668.83 | 686.56 | Close above EMA400 |

### Cycle 14 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 771.45 | 693.22 | 693.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 780.75 | 703.17 | 698.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 824.40 | 824.40 | 790.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 13:15:00 | 838.95 | 824.62 | 791.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 805.15 | 831.52 | 808.24 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 753.90 | 794.95 | 795.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 752.00 | 794.52 | 794.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 770.65 | 767.23 | 776.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 759.75 | 771.59 | 777.41 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-15 10:15:00 | 758.85 | 746.00 | 758.69 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-08-24 12:15:00 | 484.70 | 2023-08-25 10:15:00 | 478.29 | TARGET | 6.41 |
| SELL | 2023-08-24 14:15:00 | 483.65 | 2023-08-25 10:15:00 | 474.15 | TARGET | 9.50 |
| BUY | 2023-10-30 11:15:00 | 543.80 | 2023-11-03 13:15:00 | 597.21 | TARGET | 53.41 |
| SELL | 2024-06-04 09:15:00 | 839.70 | 2024-06-04 10:15:00 | 808.41 | TARGET | 31.29 |
| SELL | 2024-06-04 10:15:00 | 784.50 | 2024-06-07 15:15:00 | 845.00 | EXIT_EMA400 | -60.50 |
| SELL | 2024-07-04 10:15:00 | 837.20 | 2024-07-16 10:15:00 | 848.85 | EXIT_EMA400 | -11.65 |
| SELL | 2024-07-10 10:15:00 | 830.90 | 2024-07-16 10:15:00 | 848.85 | EXIT_EMA400 | -17.95 |
| BUY | 2024-09-20 09:15:00 | 858.00 | 2024-09-23 11:15:00 | 900.48 | TARGET | 42.48 |
| SELL | 2024-11-22 14:15:00 | 802.80 | 2024-11-25 09:15:00 | 832.80 | EXIT_EMA400 | -30.00 |
| BUY | 2024-12-23 09:15:00 | 851.80 | 2024-12-23 12:15:00 | 838.70 | EXIT_EMA400 | -13.10 |
| SELL | 2025-02-05 14:15:00 | 764.15 | 2025-02-12 09:15:00 | 693.58 | TARGET | 70.57 |
| SELL | 2025-03-21 14:15:00 | 695.65 | 2025-04-07 09:15:00 | 637.41 | TARGET | 58.24 |
| SELL | 2025-04-24 09:15:00 | 683.05 | 2025-04-25 09:15:00 | 661.72 | TARGET | 21.33 |
| BUY | 2025-07-08 13:15:00 | 838.95 | 2025-07-28 09:15:00 | 805.15 | EXIT_EMA400 | -33.80 |
| SELL | 2025-09-23 10:15:00 | 759.75 | 2025-10-15 10:15:00 | 758.85 | EXIT_EMA400 | 0.90 |
