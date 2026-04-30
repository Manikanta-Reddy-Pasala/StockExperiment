# Global Health Ltd. (MEDANTA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1120.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 5 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 2 / 6
- **Target hits / EMA400 exits:** 2 / 6
- **Total realized P&L (per unit):** -91.56
- **Avg P&L per closed trade:** -11.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 1180.55 | 1093.97 | 1093.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 1184.95 | 1102.81 | 1098.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 1106.00 | 1107.54 | 1101.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-13 14:15:00 | 1113.45 | 1107.34 | 1101.47 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 1111.90 | 1107.98 | 1101.97 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-17 10:15:00 | 1131.60 | 1108.30 | 1102.25 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1123.50 | 1110.22 | 1103.62 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 11:15:00 | 1132.55 | 1110.62 | 1103.89 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-12-20 14:15:00 | 1094.25 | 1111.80 | 1104.82 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 13:15:00 | 1063.00 | 1100.42 | 1100.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1056.45 | 1094.68 | 1097.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 1071.10 | 1067.82 | 1081.34 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 1039.80 | 1067.67 | 1080.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 12:15:00 | 1060.65 | 1058.50 | 1073.55 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-31 14:15:00 | 1039.55 | 1058.30 | 1073.30 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-02-01 13:15:00 | 1090.65 | 1057.98 | 1072.69 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 1134.15 | 1083.35 | 1083.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 1157.35 | 1090.65 | 1087.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1172.45 | 1204.64 | 1171.21 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 13:15:00 | 1191.25 | 1203.80 | 1171.44 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1199.50 | 1229.40 | 1198.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-29 10:15:00 | 1196.00 | 1228.54 | 1198.45 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 1139.60 | 1198.63 | 1198.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1130.90 | 1197.96 | 1198.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 1185.50 | 1172.30 | 1183.51 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 15:15:00 | 1300.00 | 1192.16 | 1191.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1309.20 | 1197.59 | 1194.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 1364.80 | 1370.62 | 1329.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-17 10:15:00 | 1387.20 | 1364.91 | 1332.28 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1338.00 | 1364.42 | 1335.60 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-22 13:15:00 | 1330.30 | 1364.08 | 1335.58 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 13:15:00 | 1247.50 | 1336.19 | 1336.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 09:15:00 | 1189.30 | 1333.17 | 1334.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 11:15:00 | 1260.80 | 1260.61 | 1289.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-27 10:15:00 | 1254.20 | 1260.60 | 1288.44 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-05 09:15:00 | 1227.50 | 1188.25 | 1220.82 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-12-13 14:15:00 | 1113.45 | 2024-12-20 14:15:00 | 1094.25 | EXIT_EMA400 | -19.20 |
| BUY | 2024-12-17 10:15:00 | 1131.60 | 2024-12-20 14:15:00 | 1094.25 | EXIT_EMA400 | -37.35 |
| BUY | 2024-12-19 11:15:00 | 1132.55 | 2024-12-20 14:15:00 | 1094.25 | EXIT_EMA400 | -38.30 |
| SELL | 2025-01-24 09:15:00 | 1039.80 | 2025-02-01 13:15:00 | 1090.65 | EXIT_EMA400 | -50.85 |
| SELL | 2025-01-31 14:15:00 | 1039.55 | 2025-02-01 13:15:00 | 1090.65 | EXIT_EMA400 | -51.10 |
| BUY | 2025-04-07 13:15:00 | 1191.25 | 2025-04-08 09:15:00 | 1250.67 | TARGET | 59.42 |
| BUY | 2025-09-17 10:15:00 | 1387.20 | 2025-09-22 13:15:00 | 1330.30 | EXIT_EMA400 | -56.90 |
| SELL | 2025-11-27 10:15:00 | 1254.20 | 2025-12-08 09:15:00 | 1151.48 | TARGET | 102.72 |
