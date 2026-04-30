# Aurobindo Pharma Ltd. (AUROPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1389.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 9 |
| ENTRY1 | 6 |
| ENTRY2 | 6 |
| EXIT | 6 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 6 / 6
- **Target hits / EMA400 exits:** 6 / 6
- **Total realized P&L (per unit):** 237.97
- **Avg P&L per closed trade:** 19.83

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 990.65 | 1042.04 | 1042.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 987.30 | 1040.61 | 1041.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-26 13:15:00 | 1047.20 | 1035.67 | 1038.72 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 10:15:00 | 1107.10 | 1042.03 | 1041.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-03 09:15:00 | 1125.05 | 1050.10 | 1045.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 12:15:00 | 1080.65 | 1081.86 | 1066.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-23 09:15:00 | 1087.10 | 1081.78 | 1067.08 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1193.35 | 1172.01 | 1136.59 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-04 13:15:00 | 1204.75 | 1172.33 | 1136.93 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1189.75 | 1214.50 | 1180.24 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-28 10:15:00 | 1206.80 | 1213.97 | 1180.48 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 1472.65 | 1510.79 | 1461.21 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-09-30 14:15:00 | 1455.80 | 1510.24 | 1461.19 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 14:15:00 | 1395.90 | 1452.03 | 1452.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 09:15:00 | 1366.00 | 1446.97 | 1449.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 1269.45 | 1268.78 | 1319.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-20 14:15:00 | 1240.00 | 1268.33 | 1317.55 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-31 09:15:00 | 1313.55 | 1267.92 | 1308.89 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1228.30 | 1168.23 | 1168.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 1233.20 | 1175.12 | 1171.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 09:15:00 | 1181.50 | 1182.39 | 1175.85 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 14:15:00 | 1207.30 | 1180.78 | 1175.57 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 1184.80 | 1193.22 | 1183.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-22 12:15:00 | 1179.00 | 1193.08 | 1183.95 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 13:15:00 | 1143.80 | 1178.06 | 1178.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 14:15:00 | 1138.20 | 1177.67 | 1178.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1174.10 | 1173.00 | 1175.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 11:15:00 | 1165.30 | 1172.89 | 1175.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1165.30 | 1172.89 | 1175.29 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-12 12:15:00 | 1163.50 | 1172.80 | 1175.23 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1143.40 | 1141.58 | 1155.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-30 12:15:00 | 1132.60 | 1141.39 | 1155.53 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1148.40 | 1140.10 | 1154.11 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-02 11:15:00 | 1156.90 | 1140.36 | 1154.10 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 12:15:00 | 1098.70 | 1097.76 | 1097.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1110.20 | 1098.48 | 1098.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1097.70 | 1098.84 | 1098.32 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 1095.60 | 1097.84 | 1097.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 1089.00 | 1097.75 | 1097.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1097.80 | 1097.41 | 1097.62 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1113.70 | 1097.94 | 1097.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 09:15:00 | 1124.10 | 1099.05 | 1098.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1187.00 | 1189.48 | 1160.32 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-12 13:15:00 | 1197.60 | 1186.02 | 1162.37 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-01 09:15:00 | 1173.40 | 1195.06 | 1176.25 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 1129.10 | 1173.54 | 1173.61 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1231.40 | 1173.67 | 1173.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 1239.60 | 1174.97 | 1174.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 1176.90 | 1179.23 | 1176.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-06 13:15:00 | 1184.90 | 1178.96 | 1176.51 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 1184.90 | 1178.96 | 1176.51 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-06 14:15:00 | 1191.20 | 1179.09 | 1176.59 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1189.30 | 1179.30 | 1176.72 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-09 11:15:00 | 1203.30 | 1179.70 | 1176.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-02-10 11:15:00 | 1175.00 | 1180.92 | 1177.67 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1131.40 | 1174.37 | 1174.52 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1222.50 | 1174.38 | 1174.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 13:15:00 | 1246.90 | 1189.84 | 1182.87 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-23 09:15:00 | 1087.10 | 2024-04-29 09:15:00 | 1147.17 | TARGET | 60.07 |
| BUY | 2024-06-28 10:15:00 | 1206.80 | 2024-07-05 11:15:00 | 1285.75 | TARGET | 78.95 |
| BUY | 2024-06-04 13:15:00 | 1204.75 | 2024-07-15 11:15:00 | 1408.22 | TARGET | 203.47 |
| SELL | 2024-12-20 14:15:00 | 1240.00 | 2024-12-31 09:15:00 | 1313.55 | EXIT_EMA400 | -73.55 |
| BUY | 2025-05-12 14:15:00 | 1207.30 | 2025-05-22 12:15:00 | 1179.00 | EXIT_EMA400 | -28.30 |
| SELL | 2025-06-12 11:15:00 | 1165.30 | 2025-06-17 09:15:00 | 1135.33 | TARGET | 29.97 |
| SELL | 2025-06-12 12:15:00 | 1163.50 | 2025-06-17 10:15:00 | 1128.31 | TARGET | 35.19 |
| SELL | 2025-06-30 12:15:00 | 1132.60 | 2025-07-02 11:15:00 | 1156.90 | EXIT_EMA400 | -24.30 |
| BUY | 2025-12-12 13:15:00 | 1197.60 | 2026-01-01 09:15:00 | 1173.40 | EXIT_EMA400 | -24.20 |
| BUY | 2026-02-06 13:15:00 | 1184.90 | 2026-02-10 09:15:00 | 1210.06 | TARGET | 25.16 |
| BUY | 2026-02-06 14:15:00 | 1191.20 | 2026-02-10 11:15:00 | 1175.00 | EXIT_EMA400 | -16.20 |
| BUY | 2026-02-09 11:15:00 | 1203.30 | 2026-02-10 11:15:00 | 1175.00 | EXIT_EMA400 | -28.30 |
