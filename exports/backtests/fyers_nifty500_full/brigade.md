# Brigade Enterprises Ltd. (BRIGADE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 786.30
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 7 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -82.35
- **Avg P&L per closed trade:** -9.15

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 1135.00 | 1234.66 | 1234.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 09:15:00 | 1118.70 | 1231.61 | 1233.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 15:15:00 | 1192.50 | 1190.72 | 1208.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-28 09:15:00 | 1180.75 | 1190.62 | 1208.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1200.00 | 1190.12 | 1207.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-30 10:15:00 | 1210.90 | 1191.01 | 1207.29 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 1308.35 | 1220.46 | 1220.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 1311.65 | 1223.12 | 1221.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1323.40 | 1328.58 | 1292.81 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1144.70 | 1274.43 | 1274.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1132.35 | 1241.55 | 1256.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1200.80 | 1194.31 | 1225.36 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 1294.20 | 1240.33 | 1240.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 1302.60 | 1241.44 | 1240.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1236.20 | 1250.20 | 1245.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 09:15:00 | 1326.40 | 1250.91 | 1245.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1326.40 | 1250.91 | 1245.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-26 10:15:00 | 1244.85 | 1252.14 | 1246.89 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 13:15:00 | 1155.65 | 1244.39 | 1244.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 14:15:00 | 1146.15 | 1243.41 | 1244.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1159.75 | 1140.62 | 1180.36 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 1134.90 | 1147.36 | 1178.00 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 991.00 | 994.16 | 1048.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 984.40 | 994.06 | 1048.61 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 1007.55 | 971.86 | 1011.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 1020.90 | 972.72 | 1011.72 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 1109.30 | 1026.04 | 1025.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 15:15:00 | 1118.20 | 1026.96 | 1026.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1148.60 | 1149.83 | 1106.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 1165.60 | 1149.31 | 1108.04 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-27 12:15:00 | 1108.60 | 1147.51 | 1114.10 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1032.00 | 1099.07 | 1099.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1021.10 | 1098.29 | 1098.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 963.90 | 957.36 | 994.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 10:15:00 | 953.65 | 957.73 | 994.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 944.45 | 927.63 | 959.30 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-10 11:15:00 | 940.80 | 927.76 | 959.21 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-15 13:15:00 | 958.55 | 928.56 | 956.22 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1040.60 | 973.15 | 973.04 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 10:15:00 | 944.90 | 974.40 | 974.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-17 13:15:00 | 944.30 | 973.52 | 973.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 11:15:00 | 891.20 | 888.57 | 911.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 877.85 | 889.64 | 908.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 718.35 | 690.46 | 738.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 651.90 | 690.49 | 737.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 724.90 | 686.53 | 727.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-08 11:15:00 | 727.90 | 687.31 | 727.42 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-08-28 09:15:00 | 1180.75 | 2024-08-30 10:15:00 | 1210.90 | EXIT_EMA400 | -30.15 |
| BUY | 2024-12-23 09:15:00 | 1326.40 | 2024-12-26 10:15:00 | 1244.85 | EXIT_EMA400 | -81.55 |
| SELL | 2025-02-06 09:15:00 | 1134.90 | 2025-02-14 13:15:00 | 1005.59 | TARGET | 129.31 |
| SELL | 2025-03-25 10:15:00 | 984.40 | 2025-04-22 09:15:00 | 1020.90 | EXIT_EMA400 | -36.50 |
| BUY | 2025-06-20 10:15:00 | 1165.60 | 2025-06-27 12:15:00 | 1108.60 | EXIT_EMA400 | -57.00 |
| SELL | 2025-09-16 10:15:00 | 953.65 | 2025-10-15 13:15:00 | 958.55 | EXIT_EMA400 | -4.90 |
| SELL | 2025-10-10 11:15:00 | 940.80 | 2025-10-15 13:15:00 | 958.55 | EXIT_EMA400 | -17.75 |
| SELL | 2026-01-08 10:15:00 | 877.85 | 2026-01-20 15:15:00 | 785.66 | TARGET | 92.19 |
| SELL | 2026-03-30 09:15:00 | 651.90 | 2026-04-08 11:15:00 | 727.90 | EXIT_EMA400 | -76.00 |
