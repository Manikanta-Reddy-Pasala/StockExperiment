# PNB Housing Finance Ltd. (PNBHOUSING.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1045.65
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 8 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 0 / 9
- **Target hits / EMA400 exits:** 0 / 9
- **Total realized P&L (per unit):** -233.20
- **Avg P&L per closed trade:** -25.91

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-23 13:15:00 | 699.95 | 783.91 | 784.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 698.65 | 771.27 | 777.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 711.10 | 680.39 | 716.18 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 15:15:00 | 776.65 | 735.40 | 735.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-25 09:15:00 | 782.00 | 735.87 | 735.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 10:15:00 | 756.30 | 756.38 | 747.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-08 11:15:00 | 760.60 | 756.42 | 747.49 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 751.00 | 756.76 | 748.18 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-10 10:15:00 | 747.60 | 756.67 | 748.18 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-06-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 14:15:00 | 715.40 | 746.16 | 746.24 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 15:15:00 | 761.80 | 746.38 | 746.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-11 09:15:00 | 791.00 | 746.83 | 746.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 773.70 | 779.79 | 765.50 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-26 09:15:00 | 795.10 | 779.15 | 766.13 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 779.80 | 786.91 | 774.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-10 11:15:00 | 789.85 | 786.94 | 774.67 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-19 09:15:00 | 779.15 | 791.90 | 779.64 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 854.60 | 933.88 | 934.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 846.50 | 914.41 | 920.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 910.00 | 892.34 | 907.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-08 09:15:00 | 874.60 | 897.77 | 907.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 894.45 | 882.10 | 896.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-17 09:15:00 | 891.05 | 882.19 | 896.81 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-17 11:15:00 | 898.90 | 882.45 | 896.80 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 940.25 | 850.62 | 850.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 970.10 | 857.07 | 853.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 1050.70 | 1058.36 | 1012.98 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 1074.70 | 1058.85 | 1016.09 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1062.50 | 1081.86 | 1057.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 12:15:00 | 1057.00 | 1081.22 | 1057.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 811.25 | 1039.94 | 1040.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 784.45 | 1037.40 | 1039.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 828.55 | 827.84 | 885.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 12:15:00 | 825.45 | 827.82 | 884.95 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 875.30 | 832.16 | 878.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-22 09:15:00 | 868.45 | 833.69 | 877.90 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-23 11:15:00 | 878.70 | 836.49 | 877.39 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 942.35 | 884.19 | 884.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 960.40 | 905.80 | 899.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 952.75 | 953.37 | 933.30 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 838.35 | 917.20 | 917.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 824.10 | 914.69 | 916.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 806.70 | 806.17 | 836.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 09:15:00 | 773.55 | 804.97 | 833.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 853.75 | 801.17 | 827.10 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 988.05 | 844.79 | 844.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 11:15:00 | 1011.00 | 857.45 | 851.20 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-08 11:15:00 | 760.60 | 2024-05-10 10:15:00 | 747.60 | EXIT_EMA400 | -13.00 |
| BUY | 2024-06-26 09:15:00 | 795.10 | 2024-07-19 09:15:00 | 779.15 | EXIT_EMA400 | -15.95 |
| BUY | 2024-07-10 11:15:00 | 789.85 | 2024-07-19 09:15:00 | 779.15 | EXIT_EMA400 | -10.70 |
| SELL | 2025-01-08 09:15:00 | 874.60 | 2025-01-17 11:15:00 | 898.90 | EXIT_EMA400 | -24.30 |
| SELL | 2025-01-17 09:15:00 | 891.05 | 2025-01-17 11:15:00 | 898.90 | EXIT_EMA400 | -7.85 |
| BUY | 2025-06-17 09:15:00 | 1074.70 | 2025-07-23 12:15:00 | 1057.00 | EXIT_EMA400 | -17.70 |
| SELL | 2025-09-12 12:15:00 | 825.45 | 2025-09-23 11:15:00 | 878.70 | EXIT_EMA400 | -53.25 |
| SELL | 2025-09-22 09:15:00 | 868.45 | 2025-09-23 11:15:00 | 878.70 | EXIT_EMA400 | -10.25 |
| SELL | 2026-03-30 09:15:00 | 773.55 | 2026-04-08 09:15:00 | 853.75 | EXIT_EMA400 | -80.20 |
