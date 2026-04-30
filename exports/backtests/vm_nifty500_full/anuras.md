# Anupam Rasayan India Ltd. (ANURAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1343.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 72.35
- **Avg P&L per closed trade:** 9.04

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 15:15:00 | 1015.00 | 927.60 | 927.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1020.60 | 928.52 | 927.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 15:15:00 | 991.00 | 997.15 | 972.42 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-27 12:15:00 | 998.00 | 997.12 | 972.90 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-10 11:15:00 | 989.95 | 1024.29 | 995.84 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-01-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 10:15:00 | 948.05 | 977.92 | 977.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 13:15:00 | 940.50 | 975.05 | 976.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 09:15:00 | 925.05 | 922.96 | 943.27 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-13 09:15:00 | 914.00 | 935.51 | 944.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 908.95 | 908.11 | 924.66 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-03 13:15:00 | 904.00 | 908.06 | 924.31 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 786.40 | 767.30 | 788.53 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-07-22 09:15:00 | 788.80 | 768.80 | 788.26 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 781.65 | 703.88 | 703.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 786.90 | 709.95 | 706.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 13:15:00 | 733.50 | 735.83 | 722.76 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-27 14:15:00 | 748.30 | 735.95 | 722.89 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-07 09:15:00 | 709.50 | 740.44 | 727.57 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 1075.90 | 1095.05 | 1095.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 15:15:00 | 1069.50 | 1093.19 | 1094.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 15:15:00 | 1094.40 | 1087.78 | 1091.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-10 09:15:00 | 1075.10 | 1087.65 | 1091.01 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1078.40 | 1087.24 | 1090.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-11 09:15:00 | 1072.00 | 1087.07 | 1090.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1082.10 | 1082.86 | 1087.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-17 10:15:00 | 1095.10 | 1082.98 | 1087.92 | Close above EMA400 |

### Cycle 5 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 1238.40 | 1092.27 | 1092.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 1246.40 | 1095.19 | 1093.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 15:15:00 | 1291.10 | 1299.23 | 1251.68 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-06 09:15:00 | 1314.00 | 1265.06 | 1246.97 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1273.20 | 1290.25 | 1264.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-16 13:15:00 | 1257.70 | 1289.47 | 1264.31 | Close below EMA400 |

### Cycle 6 — SELL (started 2026-03-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 13:15:00 | 1216.10 | 1257.18 | 1257.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 1208.90 | 1255.66 | 1256.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 12:15:00 | 1269.20 | 1253.03 | 1255.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-02 10:15:00 | 1240.30 | 1253.49 | 1255.30 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-02 13:15:00 | 1256.00 | 1253.34 | 1255.20 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 1286.20 | 1256.71 | 1256.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 1291.50 | 1258.57 | 1257.57 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-27 12:15:00 | 998.00 | 2024-01-01 14:15:00 | 1073.30 | TARGET | 75.30 |
| SELL | 2024-04-03 13:15:00 | 904.00 | 2024-04-18 15:15:00 | 843.06 | TARGET | 60.94 |
| SELL | 2024-03-13 09:15:00 | 914.00 | 2024-05-07 10:15:00 | 824.00 | TARGET | 90.00 |
| BUY | 2025-03-27 14:15:00 | 748.30 | 2025-04-07 09:15:00 | 709.50 | EXIT_EMA400 | -38.80 |
| SELL | 2025-11-10 09:15:00 | 1075.10 | 2025-11-17 10:15:00 | 1095.10 | EXIT_EMA400 | -20.00 |
| SELL | 2025-11-11 09:15:00 | 1072.00 | 2025-11-17 10:15:00 | 1095.10 | EXIT_EMA400 | -23.10 |
| BUY | 2026-02-06 09:15:00 | 1314.00 | 2026-02-16 13:15:00 | 1257.70 | EXIT_EMA400 | -56.30 |
| SELL | 2026-04-02 10:15:00 | 1240.30 | 2026-04-02 13:15:00 | 1256.00 | EXIT_EMA400 | -15.70 |
