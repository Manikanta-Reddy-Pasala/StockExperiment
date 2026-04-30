# Godrej Consumer Products Ltd. (GODREJCP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1067.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 11 |
| ENTRY1 | 8 |
| ENTRY2 | 6 |
| EXIT | 8 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 5 / 9
- **Target hits / EMA400 exits:** 4 / 10
- **Total realized P&L (per unit):** 132.57
- **Avg P&L per closed trade:** 9.47

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-05 10:15:00 | 1042.00 | 1002.79 | 1002.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 1047.95 | 1011.39 | 1007.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 1110.00 | 1111.20 | 1075.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 10:15:00 | 1127.90 | 1110.68 | 1077.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 1199.20 | 1221.65 | 1184.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-14 11:15:00 | 1209.70 | 1221.32 | 1184.58 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 09:15:00 | 1197.10 | 1220.30 | 1188.54 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-03-20 13:15:00 | 1210.65 | 1219.56 | 1188.79 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-04-03 09:15:00 | 1188.75 | 1220.33 | 1196.27 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 13:15:00 | 1335.35 | 1422.11 | 1422.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 14:15:00 | 1329.85 | 1421.19 | 1421.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 1263.00 | 1256.53 | 1304.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 13:15:00 | 1252.85 | 1256.68 | 1303.12 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-09 10:15:00 | 1190.85 | 1135.13 | 1188.45 | Close above EMA400 |

### Cycle 3 — BUY (started 2025-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 13:15:00 | 1196.00 | 1110.14 | 1109.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 1211.65 | 1112.87 | 1111.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1241.50 | 1251.50 | 1214.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 1268.50 | 1249.85 | 1216.02 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1224.10 | 1248.40 | 1218.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-05 14:15:00 | 1218.00 | 1247.21 | 1218.81 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1171.40 | 1206.21 | 1206.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 1165.80 | 1204.34 | 1205.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1246.00 | 1199.77 | 1202.73 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1274.50 | 1206.18 | 1205.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1277.10 | 1207.54 | 1206.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1236.60 | 1237.08 | 1224.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-22 10:15:00 | 1242.50 | 1237.13 | 1224.46 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1227.00 | 1236.95 | 1225.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 14:15:00 | 1223.50 | 1236.53 | 1225.25 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1185.10 | 1220.60 | 1220.71 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 1245.10 | 1220.92 | 1220.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 12:15:00 | 1249.90 | 1221.80 | 1221.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1223.40 | 1240.24 | 1232.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 13:15:00 | 1243.50 | 1239.35 | 1232.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1236.20 | 1242.66 | 1235.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-16 13:15:00 | 1235.00 | 1242.59 | 1235.23 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 1174.20 | 1230.22 | 1230.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 1162.80 | 1227.90 | 1229.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1171.70 | 1147.15 | 1174.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-04 11:15:00 | 1155.00 | 1148.89 | 1173.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1148.60 | 1140.02 | 1158.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-26 10:15:00 | 1146.40 | 1140.09 | 1158.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1153.20 | 1140.71 | 1158.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-27 10:15:00 | 1149.10 | 1140.80 | 1158.11 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1150.80 | 1141.06 | 1157.73 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 09:15:00 | 1133.10 | 1141.22 | 1157.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 1147.90 | 1136.45 | 1150.14 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-12 09:15:00 | 1140.50 | 1136.49 | 1150.10 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1147.50 | 1136.80 | 1150.05 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-12 13:15:00 | 1151.00 | 1136.94 | 1150.06 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1200.90 | 1159.19 | 1159.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 1206.60 | 1161.76 | 1160.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 1178.30 | 1214.75 | 1195.82 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-24 10:15:00 | 1228.80 | 1198.18 | 1192.23 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 1184.00 | 1205.10 | 1196.70 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1116.70 | 1189.01 | 1189.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1070.50 | 1187.83 | 1188.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1074.10 | 1067.39 | 1110.82 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 14:15:00 | 1062.70 | 1068.03 | 1108.62 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1102.80 | 1070.70 | 1104.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 1105.70 | 1071.05 | 1104.19 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-19 10:15:00 | 1127.90 | 2024-02-01 09:15:00 | 1279.72 | TARGET | 151.82 |
| BUY | 2024-03-14 11:15:00 | 1209.70 | 2024-04-03 09:15:00 | 1188.75 | EXIT_EMA400 | -20.95 |
| BUY | 2024-03-20 13:15:00 | 1210.65 | 2024-04-03 09:15:00 | 1188.75 | EXIT_EMA400 | -21.90 |
| SELL | 2024-11-28 13:15:00 | 1252.85 | 2024-12-09 09:15:00 | 1102.04 | TARGET | 150.81 |
| BUY | 2025-06-02 09:15:00 | 1268.50 | 2025-06-05 14:15:00 | 1218.00 | EXIT_EMA400 | -50.50 |
| BUY | 2025-07-22 10:15:00 | 1242.50 | 2025-07-24 14:15:00 | 1223.50 | EXIT_EMA400 | -19.00 |
| BUY | 2025-09-09 13:15:00 | 1243.50 | 2025-09-16 13:15:00 | 1235.00 | EXIT_EMA400 | -8.50 |
| SELL | 2025-11-27 10:15:00 | 1149.10 | 2025-12-03 09:15:00 | 1122.06 | TARGET | 27.04 |
| SELL | 2025-11-26 10:15:00 | 1146.40 | 2025-12-08 13:15:00 | 1110.45 | TARGET | 35.95 |
| SELL | 2025-11-04 11:15:00 | 1155.00 | 2025-12-12 13:15:00 | 1151.00 | EXIT_EMA400 | 4.00 |
| SELL | 2025-12-01 09:15:00 | 1133.10 | 2025-12-12 13:15:00 | 1151.00 | EXIT_EMA400 | -17.90 |
| SELL | 2025-12-12 09:15:00 | 1140.50 | 2025-12-12 13:15:00 | 1151.00 | EXIT_EMA400 | -10.50 |
| BUY | 2026-02-24 10:15:00 | 1228.80 | 2026-03-02 09:15:00 | 1184.00 | EXIT_EMA400 | -44.80 |
| SELL | 2026-04-09 14:15:00 | 1062.70 | 2026-04-17 11:15:00 | 1105.70 | EXIT_EMA400 | -43.00 |
