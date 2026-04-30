# Brigade Enterprises Ltd. (BRIGADE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 790.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -81.65
- **Avg P&L per closed trade:** -9.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 1127.15 | 1224.25 | 1224.51 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 12:15:00 | 1321.00 | 1214.83 | 1214.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1342.05 | 1242.98 | 1229.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 1323.40 | 1328.53 | 1291.58 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1141.55 | 1273.05 | 1273.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 1132.35 | 1241.74 | 1255.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 1200.80 | 1194.30 | 1224.97 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1273.90 | 1239.45 | 1239.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 14:15:00 | 1294.20 | 1240.31 | 1239.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 1237.05 | 1250.15 | 1245.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-23 09:15:00 | 1325.85 | 1250.85 | 1245.66 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1250.60 | 1251.65 | 1246.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-24 12:15:00 | 1264.90 | 1251.88 | 1246.44 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1247.00 | 1252.21 | 1246.71 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-26 10:15:00 | 1244.85 | 1252.13 | 1246.70 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 1146.15 | 1243.56 | 1244.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1122.00 | 1241.41 | 1242.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 11:15:00 | 1159.75 | 1140.64 | 1180.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-06 09:15:00 | 1134.90 | 1146.27 | 1178.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 991.00 | 993.85 | 1049.00 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 984.40 | 993.76 | 1048.68 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 1007.55 | 971.69 | 1011.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 1020.90 | 972.56 | 1011.72 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 14:15:00 | 1109.30 | 1025.92 | 1025.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 15:15:00 | 1118.20 | 1026.84 | 1026.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 10:15:00 | 1148.60 | 1149.71 | 1106.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 1165.60 | 1149.25 | 1108.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-27 12:15:00 | 1108.60 | 1147.47 | 1114.08 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1032.00 | 1099.05 | 1099.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 11:15:00 | 1021.70 | 1098.28 | 1098.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 963.90 | 957.36 | 994.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-16 10:15:00 | 953.65 | 957.73 | 994.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-10-15 13:15:00 | 958.80 | 928.66 | 956.28 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1040.60 | 973.24 | 973.11 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 944.80 | 974.18 | 974.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 932.20 | 972.71 | 973.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 11:15:00 | 891.20 | 888.71 | 911.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-08 10:15:00 | 878.40 | 889.79 | 908.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 718.35 | 690.79 | 739.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-03-30 09:15:00 | 651.90 | 690.80 | 738.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 724.85 | 686.74 | 728.26 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-09 09:15:00 | 699.85 | 688.90 | 727.94 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 725.00 | 691.46 | 727.16 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-13 10:15:00 | 727.75 | 692.77 | 727.11 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-23 09:15:00 | 1325.85 | 2024-12-26 10:15:00 | 1244.85 | EXIT_EMA400 | -81.00 |
| BUY | 2024-12-24 12:15:00 | 1264.90 | 2024-12-26 10:15:00 | 1244.85 | EXIT_EMA400 | -20.05 |
| SELL | 2025-02-06 09:15:00 | 1134.90 | 2025-02-17 09:15:00 | 1004.01 | TARGET | 130.89 |
| SELL | 2025-03-25 10:15:00 | 984.40 | 2025-04-22 09:15:00 | 1020.90 | EXIT_EMA400 | -36.50 |
| BUY | 2025-06-20 10:15:00 | 1165.60 | 2025-06-27 12:15:00 | 1108.60 | EXIT_EMA400 | -57.00 |
| SELL | 2025-09-16 10:15:00 | 953.65 | 2025-10-15 13:15:00 | 958.80 | EXIT_EMA400 | -5.15 |
| SELL | 2026-01-08 10:15:00 | 878.40 | 2026-01-20 09:15:00 | 787.50 | TARGET | 90.90 |
| SELL | 2026-03-30 09:15:00 | 651.90 | 2026-04-13 10:15:00 | 727.75 | EXIT_EMA400 | -75.85 |
| SELL | 2026-04-09 09:15:00 | 699.85 | 2026-04-13 10:15:00 | 727.75 | EXIT_EMA400 | -27.90 |
