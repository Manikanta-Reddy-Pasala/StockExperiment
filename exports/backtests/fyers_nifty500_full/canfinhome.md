# Can Fin Homes Ltd. (CANFINHOME.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 871.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -170.30
- **Avg P&L per closed trade:** -28.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 15:15:00 | 817.85 | 857.45 | 857.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 09:15:00 | 811.80 | 840.64 | 847.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 10:15:00 | 629.00 | 623.29 | 667.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-12 09:15:00 | 607.05 | 623.00 | 662.78 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-19 13:15:00 | 660.00 | 623.11 | 656.83 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 10:15:00 | 742.60 | 669.10 | 668.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-23 12:15:00 | 746.10 | 670.56 | 669.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 09:15:00 | 766.00 | 769.96 | 743.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-24 11:15:00 | 782.50 | 769.72 | 745.34 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 777.10 | 796.22 | 776.38 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 11:15:00 | 773.05 | 795.99 | 776.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 759.40 | 765.48 | 765.51 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 778.10 | 765.57 | 765.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 778.45 | 765.79 | 765.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 14:15:00 | 766.30 | 767.98 | 766.85 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 11:15:00 | 729.75 | 765.53 | 765.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 12:15:00 | 727.00 | 765.15 | 765.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 15:15:00 | 759.00 | 757.37 | 761.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 10:15:00 | 749.20 | 758.19 | 761.40 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-16 09:15:00 | 774.90 | 753.42 | 758.03 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 779.50 | 761.74 | 761.72 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 12:15:00 | 740.45 | 761.59 | 761.65 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 793.20 | 761.85 | 761.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 796.90 | 764.12 | 762.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 13:15:00 | 900.10 | 908.31 | 879.04 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-30 13:15:00 | 931.95 | 908.43 | 880.10 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 15:15:00 | 894.00 | 918.42 | 892.17 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-09 09:15:00 | 905.40 | 918.29 | 892.24 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-01-09 13:15:00 | 889.25 | 917.52 | 892.37 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 13:15:00 | 828.80 | 898.33 | 898.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 817.50 | 894.58 | 896.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 876.65 | 874.52 | 884.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-13 09:15:00 | 860.65 | 874.42 | 884.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 873.30 | 866.63 | 879.34 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-18 11:15:00 | 884.00 | 866.81 | 879.36 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-12 09:15:00 | 607.05 | 2025-03-19 13:15:00 | 660.00 | EXIT_EMA400 | -52.95 |
| BUY | 2025-06-24 11:15:00 | 782.50 | 2025-07-24 11:15:00 | 773.05 | EXIT_EMA400 | -9.45 |
| SELL | 2025-09-05 10:15:00 | 749.20 | 2025-09-16 09:15:00 | 774.90 | EXIT_EMA400 | -25.70 |
| BUY | 2025-12-30 13:15:00 | 931.95 | 2026-01-09 13:15:00 | 889.25 | EXIT_EMA400 | -42.70 |
| BUY | 2026-01-09 09:15:00 | 905.40 | 2026-01-09 13:15:00 | 889.25 | EXIT_EMA400 | -16.15 |
| SELL | 2026-03-13 09:15:00 | 860.65 | 2026-03-18 11:15:00 | 884.00 | EXIT_EMA400 | -23.35 |
