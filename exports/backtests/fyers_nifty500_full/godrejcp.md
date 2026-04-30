# Godrej Consumer Products Ltd. (GODREJCP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1069.90
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
| ENTRY1 | 7 |
| ENTRY2 | 3 |
| EXIT | 7 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 3 / 7
- **Target hits / EMA400 exits:** 3 / 7
- **Total realized P&L (per unit):** 15.46
- **Avg P&L per closed trade:** 1.55

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 1340.05 | 1428.10 | 1428.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 14:15:00 | 1329.05 | 1421.15 | 1424.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 14:15:00 | 1263.00 | 1256.22 | 1304.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 13:15:00 | 1252.95 | 1256.38 | 1303.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-09 10:15:00 | 1190.85 | 1135.06 | 1188.58 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 1199.40 | 1109.44 | 1109.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 09:15:00 | 1211.65 | 1113.02 | 1111.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 1251.40 | 1251.64 | 1214.27 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 09:15:00 | 1268.50 | 1249.84 | 1216.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1224.10 | 1248.36 | 1218.67 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-05 14:15:00 | 1218.00 | 1247.18 | 1218.81 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1171.40 | 1206.12 | 1206.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 1165.80 | 1204.26 | 1205.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1246.00 | 1199.77 | 1202.72 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 12:15:00 | 1274.40 | 1206.18 | 1205.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1277.30 | 1207.54 | 1206.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1236.60 | 1237.08 | 1224.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-22 15:15:00 | 1246.80 | 1237.33 | 1224.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1227.00 | 1236.99 | 1225.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-24 14:15:00 | 1223.50 | 1236.57 | 1225.26 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1184.80 | 1220.60 | 1220.71 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 1245.10 | 1220.93 | 1220.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 12:15:00 | 1249.90 | 1221.81 | 1221.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1223.40 | 1240.27 | 1232.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-09 13:15:00 | 1243.50 | 1239.35 | 1232.35 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1236.20 | 1242.71 | 1235.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-16 13:15:00 | 1234.90 | 1242.63 | 1235.26 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 1174.20 | 1230.20 | 1230.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 14:15:00 | 1162.80 | 1227.89 | 1229.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1171.60 | 1147.17 | 1174.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-06 09:15:00 | 1139.90 | 1149.25 | 1173.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1148.60 | 1140.00 | 1158.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-26 10:15:00 | 1146.40 | 1140.06 | 1158.38 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1153.20 | 1140.69 | 1158.16 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-27 11:15:00 | 1147.20 | 1140.84 | 1158.06 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1150.80 | 1141.08 | 1157.75 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-01 09:15:00 | 1133.10 | 1141.23 | 1157.26 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-12-11 15:15:00 | 1151.00 | 1136.50 | 1150.19 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 12:15:00 | 1200.90 | 1159.20 | 1159.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-29 11:15:00 | 1206.60 | 1161.77 | 1160.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 1178.30 | 1214.74 | 1195.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-02-24 14:15:00 | 1234.40 | 1198.34 | 1192.24 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-02 09:15:00 | 1184.00 | 1204.29 | 1196.08 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1116.70 | 1188.40 | 1188.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1070.10 | 1187.22 | 1188.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 1074.10 | 1067.29 | 1110.56 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-09 14:15:00 | 1062.70 | 1067.94 | 1108.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 1102.80 | 1070.59 | 1103.95 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-17 11:15:00 | 1105.70 | 1070.94 | 1103.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 13:15:00 | 1252.95 | 2024-12-13 09:15:00 | 1101.21 | TARGET | 151.74 |
| BUY | 2025-06-02 09:15:00 | 1268.50 | 2025-06-05 14:15:00 | 1218.00 | EXIT_EMA400 | -50.50 |
| BUY | 2025-07-22 15:15:00 | 1246.80 | 2025-07-24 14:15:00 | 1223.50 | EXIT_EMA400 | -23.30 |
| BUY | 2025-09-09 13:15:00 | 1243.50 | 2025-09-16 13:15:00 | 1234.90 | EXIT_EMA400 | -8.60 |
| SELL | 2025-11-26 10:15:00 | 1146.40 | 2025-12-08 13:15:00 | 1110.46 | TARGET | 35.94 |
| SELL | 2025-11-27 11:15:00 | 1147.20 | 2025-12-08 13:15:00 | 1114.63 | TARGET | 32.57 |
| SELL | 2025-11-06 09:15:00 | 1139.90 | 2025-12-11 15:15:00 | 1151.00 | EXIT_EMA400 | -11.10 |
| SELL | 2025-12-01 09:15:00 | 1133.10 | 2025-12-11 15:15:00 | 1151.00 | EXIT_EMA400 | -17.90 |
| BUY | 2026-02-24 14:15:00 | 1234.40 | 2026-03-02 09:15:00 | 1184.00 | EXIT_EMA400 | -50.40 |
| SELL | 2026-04-09 14:15:00 | 1062.70 | 2026-04-17 11:15:00 | 1105.70 | EXIT_EMA400 | -43.00 |
