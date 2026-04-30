# Global Health Ltd. (MEDANTA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1117.25
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 2 |
| EXIT | 6 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -202.13
- **Avg P&L per closed trade:** -25.27

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 10:15:00 | 1165.10 | 1303.26 | 1303.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 1121.35 | 1294.84 | 1299.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 11:15:00 | 1258.50 | 1252.77 | 1273.90 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-12 09:15:00 | 1229.90 | 1274.51 | 1280.50 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-31 09:15:00 | 1266.10 | 1237.53 | 1255.84 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 11:15:00 | 1178.80 | 1092.96 | 1092.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 12:15:00 | 1180.55 | 1093.83 | 1093.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 1097.30 | 1107.36 | 1100.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 15:15:00 | 1120.75 | 1107.39 | 1101.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1111.90 | 1107.93 | 1101.46 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-17 10:15:00 | 1131.60 | 1108.24 | 1101.75 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 1094.25 | 1111.74 | 1104.37 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 1063.00 | 1100.01 | 1100.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1056.45 | 1094.65 | 1097.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 1071.10 | 1067.84 | 1081.14 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 1039.80 | 1067.72 | 1080.76 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1063.45 | 1058.33 | 1073.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-03 11:15:00 | 1047.05 | 1058.19 | 1072.80 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-05 09:15:00 | 1103.85 | 1057.28 | 1071.47 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 13:15:00 | 1135.00 | 1083.84 | 1083.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 1157.35 | 1090.58 | 1087.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1172.45 | 1204.73 | 1171.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 1191.25 | 1203.89 | 1171.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1199.50 | 1229.39 | 1198.50 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-29 10:15:00 | 1196.00 | 1228.52 | 1198.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 1139.60 | 1198.60 | 1198.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1130.90 | 1197.92 | 1198.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1185.50 | 1172.33 | 1183.54 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1301.80 | 1192.21 | 1191.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1309.20 | 1197.62 | 1194.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1364.80 | 1370.67 | 1329.52 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-17 10:15:00 | 1387.20 | 1364.95 | 1332.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1338.00 | 1364.46 | 1335.65 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-22 13:15:00 | 1330.30 | 1364.12 | 1335.63 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1247.50 | 1336.15 | 1336.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1189.30 | 1333.04 | 1334.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1260.60 | 1260.50 | 1289.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 13:15:00 | 1244.20 | 1260.18 | 1287.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1227.50 | 1188.25 | 1220.80 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-07-12 09:15:00 | 1229.90 | 2024-07-31 09:15:00 | 1266.10 | EXIT_EMA400 | -36.20 |
| BUY | 2024-12-13 15:15:00 | 1120.75 | 2024-12-20 14:15:00 | 1094.25 | EXIT_EMA400 | -26.50 |
| BUY | 2024-12-17 10:15:00 | 1131.60 | 2024-12-20 14:15:00 | 1094.25 | EXIT_EMA400 | -37.35 |
| SELL | 2025-01-24 09:15:00 | 1039.80 | 2025-02-05 09:15:00 | 1103.85 | EXIT_EMA400 | -64.05 |
| SELL | 2025-02-03 11:15:00 | 1047.05 | 2025-02-05 09:15:00 | 1103.85 | EXIT_EMA400 | -56.80 |
| BUY | 2025-04-07 13:15:00 | 1191.25 | 2025-04-08 09:15:00 | 1250.22 | TARGET | 58.97 |
| BUY | 2025-09-17 10:15:00 | 1387.20 | 2025-09-22 13:15:00 | 1330.30 | EXIT_EMA400 | -56.90 |
| SELL | 2025-11-27 13:15:00 | 1244.20 | 2026-01-05 09:15:00 | 1227.50 | EXIT_EMA400 | 16.70 |
