# DLF Ltd. (DLF.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 587.15
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 132.06
- **Avg P&L per closed trade:** 18.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 776.65 | 851.85 | 852.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 763.90 | 824.43 | 836.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 13:15:00 | 806.90 | 806.15 | 823.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-22 14:15:00 | 802.80 | 806.11 | 823.61 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-25 09:15:00 | 832.80 | 806.35 | 823.56 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 869.30 | 832.44 | 832.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 871.05 | 835.22 | 833.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 828.15 | 845.79 | 839.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 09:15:00 | 852.00 | 845.72 | 839.83 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 852.00 | 845.72 | 839.83 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-23 12:15:00 | 838.70 | 845.66 | 839.88 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 11:15:00 | 805.05 | 836.34 | 836.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 804.20 | 834.99 | 835.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 764.10 | 763.47 | 787.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-10 09:15:00 | 741.20 | 763.30 | 784.56 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 706.85 | 684.32 | 714.94 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 15:15:00 | 693.00 | 687.42 | 713.66 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 685.25 | 667.59 | 690.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-24 09:15:00 | 683.05 | 669.23 | 689.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-05-02 09:15:00 | 687.20 | 668.76 | 686.33 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 771.45 | 693.21 | 692.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 780.75 | 703.14 | 698.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 824.10 | 824.36 | 790.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 13:15:00 | 838.95 | 824.58 | 791.60 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 805.35 | 831.50 | 808.20 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 754.00 | 794.89 | 795.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 752.00 | 794.46 | 794.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 11:15:00 | 770.65 | 767.23 | 776.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-23 10:15:00 | 759.75 | 771.61 | 777.40 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 758.65 | 745.98 | 758.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-15 11:15:00 | 763.45 | 746.15 | 758.70 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-22 14:15:00 | 802.80 | 2024-11-25 09:15:00 | 832.80 | EXIT_EMA400 | -30.00 |
| BUY | 2024-12-23 09:15:00 | 852.00 | 2024-12-23 12:15:00 | 838.70 | EXIT_EMA400 | -13.30 |
| SELL | 2025-02-10 09:15:00 | 741.20 | 2025-04-07 09:15:00 | 611.13 | TARGET | 130.07 |
| SELL | 2025-03-25 15:15:00 | 693.00 | 2025-04-07 09:15:00 | 631.01 | TARGET | 61.99 |
| SELL | 2025-04-24 09:15:00 | 683.05 | 2025-04-25 09:15:00 | 662.45 | TARGET | 20.60 |
| BUY | 2025-07-08 13:15:00 | 838.95 | 2025-07-28 09:15:00 | 805.35 | EXIT_EMA400 | -33.60 |
| SELL | 2025-09-23 10:15:00 | 759.75 | 2025-10-15 11:15:00 | 763.45 | EXIT_EMA400 | -3.70 |
