# Aegis Logistics Ltd. (AEGISLOG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 703.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** 32.33
- **Avg P&L per closed trade:** 4.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 15:15:00 | 745.00 | 787.76 | 787.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 10:15:00 | 742.65 | 786.89 | 787.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 11:15:00 | 747.00 | 739.05 | 757.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-22 14:15:00 | 725.90 | 739.16 | 756.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-29 09:15:00 | 757.50 | 740.88 | 755.12 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 09:15:00 | 771.95 | 765.66 | 765.63 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 11:15:00 | 758.15 | 765.57 | 765.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 13:15:00 | 772.00 | 765.65 | 765.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-14 09:15:00 | 809.40 | 766.13 | 765.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 806.95 | 811.35 | 793.91 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 15:15:00 | 840.00 | 796.58 | 790.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 790.75 | 796.69 | 790.44 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-27 12:15:00 | 826.35 | 798.82 | 792.12 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-01-13 13:15:00 | 808.05 | 834.10 | 814.39 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 15:15:00 | 676.00 | 802.60 | 802.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 636.30 | 800.95 | 801.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 759.25 | 750.65 | 772.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 731.05 | 763.16 | 775.92 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 769.60 | 763.27 | 775.85 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-12 13:15:00 | 758.85 | 763.26 | 775.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-12 14:15:00 | 796.00 | 763.58 | 775.81 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 15:15:00 | 902.00 | 765.18 | 765.04 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 746.90 | 796.98 | 797.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 742.30 | 778.50 | 786.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 732.75 | 729.40 | 748.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 12:15:00 | 705.75 | 729.44 | 746.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 741.65 | 713.35 | 730.04 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 764.90 | 742.97 | 742.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 766.45 | 743.21 | 743.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 745.45 | 745.48 | 744.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 769.60 | 745.78 | 744.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 12:15:00 | 742.05 | 746.00 | 744.54 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 739.20 | 772.47 | 772.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 725.55 | 771.67 | 772.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 754.05 | 739.83 | 751.91 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-16 09:15:00 | 719.00 | 738.49 | 748.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 12:15:00 | 741.65 | 710.71 | 729.29 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 699.90 | 672.79 | 672.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 15:15:00 | 703.90 | 673.89 | 673.26 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-22 14:15:00 | 725.90 | 2024-10-29 09:15:00 | 757.50 | EXIT_EMA400 | -31.60 |
| BUY | 2024-12-27 12:15:00 | 826.35 | 2025-01-07 11:15:00 | 929.05 | TARGET | 102.70 |
| BUY | 2024-12-20 15:15:00 | 840.00 | 2025-01-07 15:15:00 | 989.43 | TARGET | 149.43 |
| SELL | 2025-02-12 09:15:00 | 731.05 | 2025-02-12 14:15:00 | 796.00 | EXIT_EMA400 | -64.95 |
| SELL | 2025-02-12 13:15:00 | 758.85 | 2025-02-12 14:15:00 | 796.00 | EXIT_EMA400 | -37.15 |
| SELL | 2025-08-26 12:15:00 | 705.75 | 2025-09-15 09:15:00 | 741.65 | EXIT_EMA400 | -35.90 |
| BUY | 2025-09-29 09:15:00 | 769.60 | 2025-09-29 12:15:00 | 742.05 | EXIT_EMA400 | -27.55 |
| SELL | 2026-01-16 09:15:00 | 719.00 | 2026-01-30 12:15:00 | 741.65 | EXIT_EMA400 | -22.65 |
