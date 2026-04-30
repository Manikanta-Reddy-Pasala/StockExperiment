# Jubilant Pharmova Ltd. (JUBLPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 926.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 7 |
| ENTRY2 | 6 |
| EXIT | 7 |

## P&L

- **Trades closed:** 13
- **Trades open at end:** 0
- **Winners / losers:** 5 / 8
- **Target hits / EMA400 exits:** 2 / 11
- **Total realized P&L (per unit):** -189.70
- **Avg P&L per closed trade:** -14.59

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 11:15:00 | 345.80 | 418.75 | 419.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-25 12:15:00 | 338.40 | 417.95 | 418.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 14:15:00 | 400.85 | 399.96 | 407.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 11:15:00 | 398.50 | 400.05 | 407.68 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 406.05 | 400.02 | 406.90 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-10 12:15:00 | 407.85 | 400.10 | 406.90 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 13:15:00 | 434.70 | 411.71 | 411.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 13:15:00 | 442.95 | 415.01 | 413.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 533.60 | 538.23 | 502.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-24 12:15:00 | 543.85 | 538.28 | 503.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 549.55 | 573.05 | 548.05 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-02-29 12:15:00 | 557.50 | 572.71 | 548.13 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 09:15:00 | 566.55 | 575.09 | 554.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-12 13:15:00 | 578.00 | 574.76 | 554.71 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 557.05 | 574.24 | 554.94 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-03-13 13:15:00 | 549.15 | 573.88 | 554.95 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 1040.75 | 1113.91 | 1114.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 1036.20 | 1112.41 | 1113.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 10:15:00 | 1031.00 | 984.68 | 1027.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 966.75 | 993.15 | 1026.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 990.90 | 986.23 | 1017.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-18 09:15:00 | 957.55 | 986.04 | 1017.32 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 1000.15 | 985.32 | 1016.19 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-19 15:15:00 | 969.00 | 984.99 | 1014.81 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-03 09:15:00 | 965.90 | 910.02 | 944.28 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 1065.85 | 931.38 | 931.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1120.20 | 937.39 | 934.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 15:15:00 | 1170.00 | 1171.64 | 1122.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-25 14:15:00 | 1181.50 | 1171.65 | 1124.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1150.90 | 1172.73 | 1127.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-30 11:15:00 | 1221.90 | 1173.53 | 1129.11 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-08-04 09:15:00 | 1123.30 | 1176.03 | 1134.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 1054.80 | 1112.94 | 1113.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1044.20 | 1112.26 | 1112.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1133.00 | 1086.59 | 1097.87 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-19 15:15:00 | 1090.00 | 1100.56 | 1102.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-22 09:15:00 | 1105.40 | 1100.61 | 1102.85 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1122.00 | 1098.53 | 1098.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 13:15:00 | 1131.00 | 1099.83 | 1099.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 12:15:00 | 1095.50 | 1102.55 | 1100.60 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-03 10:15:00 | 1140.90 | 1102.96 | 1100.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 1117.70 | 1116.47 | 1108.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-12 10:15:00 | 1124.50 | 1116.60 | 1108.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-11-19 14:15:00 | 1107.90 | 1121.94 | 1113.27 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 1076.60 | 1106.55 | 1106.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 14:15:00 | 1066.20 | 1104.21 | 1105.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 14:15:00 | 1092.30 | 1092.29 | 1098.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-12 09:15:00 | 1084.80 | 1092.21 | 1098.56 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-19 14:15:00 | 1095.00 | 1084.27 | 1093.12 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-07 11:15:00 | 398.50 | 2023-11-10 12:15:00 | 407.85 | EXIT_EMA400 | -9.35 |
| BUY | 2024-02-29 12:15:00 | 557.50 | 2024-03-05 13:15:00 | 585.62 | TARGET | 28.12 |
| BUY | 2024-01-24 12:15:00 | 543.85 | 2024-03-13 13:15:00 | 549.15 | EXIT_EMA400 | 5.30 |
| BUY | 2024-03-12 13:15:00 | 578.00 | 2024-03-13 13:15:00 | 549.15 | EXIT_EMA400 | -28.85 |
| SELL | 2025-02-11 09:15:00 | 966.75 | 2025-04-03 09:15:00 | 965.90 | EXIT_EMA400 | 0.85 |
| SELL | 2025-02-18 09:15:00 | 957.55 | 2025-04-03 09:15:00 | 965.90 | EXIT_EMA400 | -8.35 |
| SELL | 2025-02-19 15:15:00 | 969.00 | 2025-04-03 09:15:00 | 965.90 | EXIT_EMA400 | 3.10 |
| BUY | 2025-07-25 14:15:00 | 1181.50 | 2025-08-04 09:15:00 | 1123.30 | EXIT_EMA400 | -58.20 |
| BUY | 2025-07-30 11:15:00 | 1221.90 | 2025-08-04 09:15:00 | 1123.30 | EXIT_EMA400 | -98.60 |
| SELL | 2025-09-19 15:15:00 | 1090.00 | 2025-09-22 09:15:00 | 1105.40 | EXIT_EMA400 | -15.40 |
| BUY | 2025-11-03 10:15:00 | 1140.90 | 2025-11-19 14:15:00 | 1107.90 | EXIT_EMA400 | -33.00 |
| BUY | 2025-11-12 10:15:00 | 1124.50 | 2025-11-19 14:15:00 | 1107.90 | EXIT_EMA400 | -16.60 |
| SELL | 2025-12-12 09:15:00 | 1084.80 | 2025-12-17 13:15:00 | 1043.52 | TARGET | 41.28 |
