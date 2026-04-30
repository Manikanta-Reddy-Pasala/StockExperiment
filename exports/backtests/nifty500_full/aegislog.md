# Aegis Logistics Ltd. (AEGISLOG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-06-07 09:15:00 → 2026-04-30 15:30:00 (3261 bars)
- **Last close:** 700.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -78.41
- **Avg P&L per closed trade:** -8.71

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 09:15:00 | 852.45 | 773.61 | 773.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 14:15:00 | 864.35 | 792.34 | 783.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 15:15:00 | 806.20 | 811.12 | 795.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-20 15:15:00 | 835.00 | 796.43 | 791.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 797.85 | 796.49 | 791.71 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-23 13:15:00 | 808.45 | 796.61 | 791.85 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-23 14:15:00 | 790.75 | 796.56 | 791.84 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 675.80 | 803.84 | 803.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 636.30 | 800.91 | 802.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 759.65 | 754.76 | 776.04 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-12 09:15:00 | 731.05 | 766.00 | 778.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 769.60 | 766.06 | 778.57 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-12 13:15:00 | 758.85 | 765.99 | 778.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-12 14:15:00 | 796.05 | 766.29 | 778.50 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 835.60 | 766.97 | 766.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 843.95 | 782.15 | 777.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 773.25 | 783.84 | 778.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-09 11:15:00 | 795.65 | 783.91 | 778.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 795.65 | 783.91 | 778.50 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-12 09:15:00 | 818.50 | 784.65 | 779.00 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-27 13:15:00 | 801.80 | 822.32 | 803.31 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 746.90 | 796.94 | 797.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 10:15:00 | 742.30 | 778.48 | 786.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 11:15:00 | 732.75 | 729.39 | 748.59 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-26 12:15:00 | 706.05 | 729.49 | 746.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 741.65 | 713.38 | 730.07 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 764.90 | 743.02 | 742.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 766.45 | 743.26 | 743.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 745.00 | 745.52 | 744.28 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-29 09:15:00 | 769.60 | 745.84 | 744.46 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 12:15:00 | 742.05 | 746.05 | 744.59 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 739.20 | 772.47 | 772.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 725.55 | 771.67 | 772.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 09:15:00 | 753.50 | 739.80 | 751.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-14 12:15:00 | 720.60 | 739.25 | 748.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-30 12:15:00 | 741.65 | 710.80 | 729.34 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-20 15:15:00 | 835.00 | 2024-12-23 14:15:00 | 790.75 | EXIT_EMA400 | -44.25 |
| BUY | 2024-12-23 13:15:00 | 808.45 | 2024-12-23 14:15:00 | 790.75 | EXIT_EMA400 | -17.70 |
| SELL | 2025-02-12 09:15:00 | 731.05 | 2025-02-12 14:15:00 | 796.05 | EXIT_EMA400 | -65.00 |
| SELL | 2025-02-12 13:15:00 | 758.85 | 2025-02-12 14:15:00 | 796.05 | EXIT_EMA400 | -37.20 |
| BUY | 2025-05-09 11:15:00 | 795.65 | 2025-05-15 10:15:00 | 847.11 | TARGET | 51.46 |
| BUY | 2025-05-12 09:15:00 | 818.50 | 2025-05-20 09:15:00 | 936.99 | TARGET | 118.49 |
| SELL | 2025-08-26 12:15:00 | 706.05 | 2025-09-15 09:15:00 | 741.65 | EXIT_EMA400 | -35.60 |
| BUY | 2025-09-29 09:15:00 | 769.60 | 2025-09-29 12:15:00 | 742.05 | EXIT_EMA400 | -27.55 |
| SELL | 2026-01-14 12:15:00 | 720.60 | 2026-01-30 12:15:00 | 741.65 | EXIT_EMA400 | -21.05 |
