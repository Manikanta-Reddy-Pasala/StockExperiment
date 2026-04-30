# Zensar Technolgies Ltd. (ZENSARTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 514.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -105.88
- **Avg P&L per closed trade:** -15.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 671.80 | 745.01 | 745.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 660.80 | 740.03 | 742.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 723.35 | 715.37 | 727.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-15 14:15:00 | 708.00 | 715.46 | 727.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 699.05 | 701.28 | 714.20 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-04 12:15:00 | 716.40 | 701.57 | 714.15 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 754.25 | 719.95 | 719.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 759.15 | 721.38 | 720.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 765.20 | 766.53 | 749.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 14:15:00 | 772.65 | 766.59 | 749.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 765.20 | 766.58 | 749.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-23 11:15:00 | 742.00 | 766.12 | 749.24 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-03-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 12:15:00 | 745.90 | 793.72 | 793.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 738.05 | 791.68 | 792.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 14:15:00 | 688.90 | 687.89 | 719.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-22 14:15:00 | 676.80 | 687.81 | 718.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 713.20 | 689.59 | 716.30 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-28 12:15:00 | 726.65 | 690.32 | 716.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 783.80 | 727.87 | 727.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 861.95 | 730.37 | 729.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 832.00 | 839.88 | 815.92 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 10:15:00 | 846.05 | 835.11 | 815.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 13:15:00 | 817.60 | 834.02 | 818.06 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 800.00 | 809.62 | 809.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 11:15:00 | 796.20 | 808.65 | 809.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 814.65 | 807.65 | 808.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 11:15:00 | 800.35 | 808.05 | 808.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-25 09:15:00 | 814.65 | 807.63 | 808.51 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 853.95 | 805.34 | 805.33 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 759.20 | 806.04 | 806.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 13:15:00 | 757.00 | 805.55 | 805.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 793.30 | 789.54 | 796.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 15:15:00 | 777.05 | 789.10 | 795.98 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 783.30 | 783.36 | 791.80 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 13:15:00 | 781.90 | 783.35 | 791.75 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 799.50 | 783.40 | 791.57 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-15 14:15:00 | 708.00 | 2024-10-22 13:15:00 | 650.58 | TARGET | 57.42 |
| BUY | 2024-12-20 14:15:00 | 772.65 | 2024-12-23 11:15:00 | 742.00 | EXIT_EMA400 | -30.65 |
| SELL | 2025-04-22 14:15:00 | 676.80 | 2025-04-28 12:15:00 | 726.65 | EXIT_EMA400 | -49.85 |
| BUY | 2025-07-16 10:15:00 | 846.05 | 2025-07-22 13:15:00 | 817.60 | EXIT_EMA400 | -28.45 |
| SELL | 2025-08-22 11:15:00 | 800.35 | 2025-08-25 09:15:00 | 814.65 | EXIT_EMA400 | -14.30 |
| SELL | 2025-10-13 15:15:00 | 777.05 | 2025-10-23 09:15:00 | 799.50 | EXIT_EMA400 | -22.45 |
| SELL | 2025-10-20 13:15:00 | 781.90 | 2025-10-23 09:15:00 | 799.50 | EXIT_EMA400 | -17.60 |
