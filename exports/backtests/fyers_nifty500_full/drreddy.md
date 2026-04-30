# Dr. Reddy's Laboratories Ltd. (DRREDDY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1325.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 6 |
| ENTRY2 | 1 |
| EXIT | 6 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 4 / 3
- **Target hits / EMA400 exits:** 4 / 3
- **Total realized P&L (per unit):** 261.20
- **Avg P&L per closed trade:** 37.31

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1271.25 | 1330.61 | 1330.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 09:15:00 | 1242.95 | 1327.50 | 1329.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 10:15:00 | 1320.60 | 1311.14 | 1319.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-07 09:15:00 | 1287.10 | 1310.47 | 1319.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-16 12:15:00 | 1270.65 | 1245.25 | 1267.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1389.70 | 1283.63 | 1283.16 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 12:15:00 | 1182.00 | 1299.54 | 1300.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1172.25 | 1238.86 | 1261.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 11:15:00 | 1164.55 | 1162.32 | 1202.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 09:15:00 | 1156.45 | 1169.02 | 1198.53 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 1170.00 | 1164.09 | 1191.15 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-03 11:15:00 | 1159.95 | 1164.11 | 1190.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 13:15:00 | 1173.90 | 1148.70 | 1173.94 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-21 14:15:00 | 1177.80 | 1148.99 | 1173.96 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 11:15:00 | 1232.10 | 1182.48 | 1182.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1240.90 | 1184.73 | 1183.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 1278.40 | 1294.54 | 1259.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 14:15:00 | 1307.30 | 1291.33 | 1262.97 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1256.90 | 1290.18 | 1265.49 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 1188.80 | 1257.75 | 1257.90 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1272.80 | 1255.51 | 1255.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1274.40 | 1255.70 | 1255.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 1256.70 | 1257.31 | 1256.44 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-09-04 12:15:00 | 1269.90 | 1257.87 | 1256.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 1258.60 | 1257.88 | 1256.78 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-09-04 14:15:00 | 1252.80 | 1257.83 | 1256.76 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1251.50 | 1265.94 | 1265.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1250.10 | 1265.79 | 1265.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 1268.60 | 1259.68 | 1262.56 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1288.80 | 1265.08 | 1265.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1293.90 | 1266.73 | 1265.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 1252.00 | 1267.38 | 1266.26 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1199.10 | 1265.09 | 1265.13 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 15:15:00 | 1266.00 | 1252.56 | 1252.55 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 1247.10 | 1252.53 | 1252.54 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1259.60 | 1252.60 | 1252.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1261.40 | 1252.69 | 1252.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1259.50 | 1263.96 | 1259.09 | EMA200 retest candle locked |

### Cycle 13 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1210.20 | 1256.67 | 1256.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 1204.20 | 1254.40 | 1255.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 09:15:00 | 1243.50 | 1225.87 | 1239.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-28 10:15:00 | 1224.00 | 1227.53 | 1239.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1221.90 | 1221.12 | 1233.83 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 1236.10 | 1221.37 | 1233.83 | Close above EMA400 |

### Cycle 14 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1286.00 | 1241.91 | 1241.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 10:15:00 | 1288.00 | 1245.82 | 1243.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 12:15:00 | 1283.30 | 1283.55 | 1268.00 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-20 13:15:00 | 1302.30 | 1283.75 | 1270.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-23 09:15:00 | 1270.10 | 1283.89 | 1270.91 | Close below EMA400 |

### Cycle 15 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1187.90 | 1263.24 | 1263.25 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 09:15:00 | 1340.40 | 1258.95 | 1258.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 10:15:00 | 1359.00 | 1259.94 | 1259.23 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-07 09:15:00 | 1287.10 | 2024-11-18 09:15:00 | 1190.37 | TARGET | 96.73 |
| SELL | 2025-03-26 09:15:00 | 1156.45 | 2025-04-07 09:15:00 | 1030.20 | TARGET | 126.25 |
| SELL | 2025-04-03 11:15:00 | 1159.95 | 2025-04-07 09:15:00 | 1067.13 | TARGET | 92.82 |
| BUY | 2025-07-04 14:15:00 | 1307.30 | 2025-07-10 09:15:00 | 1256.90 | EXIT_EMA400 | -50.40 |
| BUY | 2025-09-04 12:15:00 | 1269.90 | 2025-09-04 14:15:00 | 1252.80 | EXIT_EMA400 | -17.10 |
| SELL | 2026-01-28 10:15:00 | 1224.00 | 2026-02-01 15:15:00 | 1178.90 | TARGET | 45.10 |
| BUY | 2026-03-20 13:15:00 | 1302.30 | 2026-03-23 09:15:00 | 1270.10 | EXIT_EMA400 | -32.20 |
