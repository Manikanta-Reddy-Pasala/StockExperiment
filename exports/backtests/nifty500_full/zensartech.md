# Zensar Technolgies Ltd. (ZENSARTECH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 513.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 3 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -157.52
- **Avg P&L per closed trade:** -19.69

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 530.00 | 555.32 | 555.34 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-12 15:15:00 | 562.80 | 553.87 | 553.84 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 09:15:00 | 544.50 | 553.77 | 553.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 11:15:00 | 532.65 | 553.46 | 553.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 14:15:00 | 552.50 | 552.07 | 552.92 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 583.55 | 553.83 | 553.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 590.35 | 556.86 | 555.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 14:15:00 | 590.05 | 591.55 | 577.88 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-26 09:15:00 | 637.85 | 587.50 | 578.33 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-09 09:15:00 | 587.00 | 600.67 | 588.18 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 674.95 | 744.33 | 744.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 670.05 | 743.59 | 744.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 14:15:00 | 722.75 | 715.37 | 727.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-15 14:15:00 | 708.05 | 715.46 | 727.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 699.05 | 701.33 | 714.27 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-04 12:15:00 | 716.00 | 701.62 | 714.22 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 754.25 | 720.02 | 719.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 759.20 | 721.45 | 720.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 13:15:00 | 765.20 | 766.59 | 749.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 14:15:00 | 772.65 | 766.65 | 749.20 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-12-23 11:15:00 | 741.60 | 766.17 | 749.31 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 741.00 | 790.27 | 790.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 738.05 | 789.75 | 790.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 14:15:00 | 688.90 | 687.57 | 718.85 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-22 14:15:00 | 676.80 | 687.52 | 717.75 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 713.65 | 689.34 | 715.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-28 12:15:00 | 726.65 | 690.08 | 715.39 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 787.35 | 727.30 | 727.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 861.30 | 730.35 | 728.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 832.00 | 839.88 | 815.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-16 10:15:00 | 846.10 | 835.11 | 815.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-22 13:15:00 | 817.10 | 834.03 | 817.98 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 802.10 | 809.51 | 809.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 799.50 | 809.41 | 809.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 814.15 | 807.71 | 808.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-22 11:15:00 | 800.35 | 808.10 | 808.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-25 09:15:00 | 814.65 | 807.68 | 808.50 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 12:15:00 | 853.95 | 805.38 | 805.35 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 759.20 | 806.07 | 806.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 13:15:00 | 757.00 | 805.58 | 805.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 09:15:00 | 793.30 | 789.47 | 796.61 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-13 15:15:00 | 777.75 | 789.09 | 795.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 783.30 | 783.39 | 791.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-20 13:15:00 | 781.90 | 783.37 | 791.76 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 799.50 | 783.44 | 791.59 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-26 09:15:00 | 637.85 | 2024-05-09 09:15:00 | 587.00 | EXIT_EMA400 | -50.85 |
| SELL | 2024-10-15 14:15:00 | 708.05 | 2024-10-22 13:15:00 | 651.17 | TARGET | 56.88 |
| BUY | 2024-12-20 14:15:00 | 772.65 | 2024-12-23 11:15:00 | 741.60 | EXIT_EMA400 | -31.05 |
| SELL | 2025-04-22 14:15:00 | 676.80 | 2025-04-28 12:15:00 | 726.65 | EXIT_EMA400 | -49.85 |
| BUY | 2025-07-16 10:15:00 | 846.10 | 2025-07-22 13:15:00 | 817.10 | EXIT_EMA400 | -29.00 |
| SELL | 2025-08-22 11:15:00 | 800.35 | 2025-08-25 09:15:00 | 814.65 | EXIT_EMA400 | -14.30 |
| SELL | 2025-10-13 15:15:00 | 777.75 | 2025-10-23 09:15:00 | 799.50 | EXIT_EMA400 | -21.75 |
| SELL | 2025-10-20 13:15:00 | 781.90 | 2025-10-23 09:15:00 | 799.50 | EXIT_EMA400 | -17.60 |
