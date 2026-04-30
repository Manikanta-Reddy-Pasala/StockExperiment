# PNB Housing Finance Ltd. (PNBHOUSING.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1042.05
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 4 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -193.00
- **Avg P&L per closed trade:** -32.17

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 15:15:00 | 851.00 | 933.93 | 933.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 12:15:00 | 847.00 | 914.36 | 920.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 910.45 | 892.30 | 906.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-08 09:15:00 | 875.10 | 897.61 | 907.59 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 894.45 | 881.95 | 896.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-17 09:15:00 | 891.05 | 882.04 | 896.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 892.50 | 882.14 | 896.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-17 11:15:00 | 898.90 | 882.31 | 896.66 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 13:15:00 | 940.40 | 850.65 | 850.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 970.10 | 857.09 | 853.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 10:15:00 | 1050.70 | 1058.32 | 1012.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-17 09:15:00 | 1074.70 | 1058.83 | 1016.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1062.50 | 1081.85 | 1057.05 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-23 12:15:00 | 1057.00 | 1081.21 | 1057.10 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 15:15:00 | 811.15 | 1039.94 | 1040.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 784.15 | 1037.39 | 1039.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 11:15:00 | 828.75 | 827.86 | 885.26 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-12 12:15:00 | 825.50 | 827.84 | 884.96 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 875.30 | 832.19 | 878.06 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-22 09:15:00 | 868.45 | 833.72 | 877.92 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-23 11:15:00 | 878.70 | 836.54 | 877.42 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 10:15:00 | 942.35 | 884.26 | 884.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 960.40 | 905.80 | 899.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 13:15:00 | 952.75 | 953.30 | 933.27 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 838.35 | 917.18 | 917.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 824.00 | 914.68 | 916.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 806.70 | 805.63 | 835.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-30 09:15:00 | 773.55 | 804.55 | 833.08 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-08 09:15:00 | 853.75 | 800.83 | 826.33 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 979.70 | 844.90 | 844.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 09:15:00 | 987.85 | 847.65 | 845.65 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-08 09:15:00 | 875.10 | 2025-01-17 11:15:00 | 898.90 | EXIT_EMA400 | -23.80 |
| SELL | 2025-01-17 09:15:00 | 891.05 | 2025-01-17 11:15:00 | 898.90 | EXIT_EMA400 | -7.85 |
| BUY | 2025-06-17 09:15:00 | 1074.70 | 2025-07-23 12:15:00 | 1057.00 | EXIT_EMA400 | -17.70 |
| SELL | 2025-09-12 12:15:00 | 825.50 | 2025-09-23 11:15:00 | 878.70 | EXIT_EMA400 | -53.20 |
| SELL | 2025-09-22 09:15:00 | 868.45 | 2025-09-23 11:15:00 | 878.70 | EXIT_EMA400 | -10.25 |
| SELL | 2026-03-30 09:15:00 | 773.55 | 2026-04-08 09:15:00 | 853.75 | EXIT_EMA400 | -80.20 |
