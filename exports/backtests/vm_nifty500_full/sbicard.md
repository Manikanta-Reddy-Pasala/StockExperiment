# SBI Cards and Payment Services Ltd. (SBICARD.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 643.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 29.92
- **Avg P&L per closed trade:** 3.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 14:15:00 | 749.30 | 726.42 | 726.33 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 09:15:00 | 715.65 | 726.20 | 726.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 713.70 | 725.97 | 726.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 09:15:00 | 713.35 | 708.76 | 715.14 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2024-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 13:15:00 | 730.50 | 718.73 | 718.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 732.00 | 718.87 | 718.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 714.95 | 720.52 | 719.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-08 12:15:00 | 732.60 | 719.73 | 719.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 732.60 | 719.73 | 719.36 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-08 13:15:00 | 735.00 | 719.88 | 719.44 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 727.80 | 725.78 | 722.82 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-18 12:15:00 | 730.40 | 725.82 | 722.86 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-19 10:15:00 | 719.30 | 725.94 | 722.99 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 12:15:00 | 705.30 | 721.66 | 721.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 13:15:00 | 700.35 | 721.45 | 721.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 712.55 | 712.42 | 716.35 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-21 12:15:00 | 709.55 | 712.39 | 716.30 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 12:15:00 | 714.30 | 712.37 | 716.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-22 13:15:00 | 716.15 | 712.40 | 716.15 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 09:15:00 | 757.00 | 719.04 | 718.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 10:15:00 | 766.75 | 719.51 | 719.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 10:15:00 | 769.00 | 770.30 | 753.57 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 14:15:00 | 705.40 | 745.41 | 745.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 696.90 | 742.29 | 743.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 10:15:00 | 702.00 | 701.06 | 715.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 685.80 | 711.67 | 715.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 703.05 | 698.66 | 707.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-03 11:15:00 | 708.25 | 698.80 | 707.30 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 751.80 | 713.68 | 713.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 15:15:00 | 752.75 | 714.07 | 713.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 11:15:00 | 841.50 | 844.05 | 818.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-11 10:15:00 | 850.70 | 841.98 | 820.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-01 11:15:00 | 924.30 | 954.51 | 925.77 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 12:15:00 | 840.10 | 913.84 | 914.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 837.50 | 913.08 | 913.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 14:15:00 | 819.65 | 818.64 | 844.14 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 891.00 | 858.04 | 857.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 904.00 | 858.84 | 858.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 898.00 | 899.93 | 883.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-28 14:15:00 | 906.50 | 899.80 | 884.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-30 14:15:00 | 885.35 | 900.44 | 885.71 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 870.35 | 879.85 | 879.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 860.15 | 879.47 | 879.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 882.55 | 877.39 | 878.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-15 09:15:00 | 868.25 | 877.04 | 878.31 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 870.00 | 869.71 | 874.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 11:15:00 | 878.70 | 869.86 | 874.16 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-07-08 12:15:00 | 732.60 | 2024-07-19 10:15:00 | 719.30 | EXIT_EMA400 | -13.30 |
| BUY | 2024-07-08 13:15:00 | 735.00 | 2024-07-19 10:15:00 | 719.30 | EXIT_EMA400 | -15.70 |
| BUY | 2024-07-18 12:15:00 | 730.40 | 2024-07-19 10:15:00 | 719.30 | EXIT_EMA400 | -11.10 |
| SELL | 2024-08-21 12:15:00 | 709.55 | 2024-08-22 13:15:00 | 716.15 | EXIT_EMA400 | -6.60 |
| SELL | 2024-12-20 14:15:00 | 685.80 | 2025-01-03 11:15:00 | 708.25 | EXIT_EMA400 | -22.45 |
| BUY | 2025-04-11 10:15:00 | 850.70 | 2025-06-04 09:15:00 | 940.72 | TARGET | 90.02 |
| BUY | 2025-10-28 14:15:00 | 906.50 | 2025-10-30 14:15:00 | 885.35 | EXIT_EMA400 | -21.15 |
| SELL | 2025-12-15 09:15:00 | 868.25 | 2025-12-17 09:15:00 | 838.06 | TARGET | 30.19 |
