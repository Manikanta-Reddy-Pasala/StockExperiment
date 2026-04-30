# Nestle India Ltd. (NESTLEIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1458.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -110.90
- **Avg P&L per closed trade:** -13.86

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 15:15:00 | 1126.62 | 1113.79 | 1113.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 1136.95 | 1115.03 | 1114.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 1118.29 | 1119.89 | 1117.11 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-04 09:15:00 | 1133.18 | 1119.89 | 1117.21 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1258.80 | 1278.70 | 1250.07 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-01-23 09:15:00 | 1240.07 | 1275.55 | 1250.38 | Close below EMA400 |

### Cycle 2 — SELL (started 2024-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 09:15:00 | 1264.12 | 1265.47 | 1265.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 12:15:00 | 1247.40 | 1265.11 | 1265.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 12:15:00 | 1275.00 | 1264.62 | 1265.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-03 11:15:00 | 1233.57 | 1262.50 | 1263.85 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1257.50 | 1258.91 | 1261.90 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-05-07 13:15:00 | 1252.60 | 1258.80 | 1261.80 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-05-08 11:15:00 | 1262.20 | 1258.65 | 1261.65 | Close above EMA400 |

### Cycle 3 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 1255.03 | 1252.19 | 1252.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 1260.43 | 1252.31 | 1252.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 09:15:00 | 1264.50 | 1282.87 | 1271.79 | EMA200 retest candle locked |

### Cycle 4 — SELL (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-07 10:15:00 | 1257.65 | 1263.71 | 1263.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 09:15:00 | 1251.50 | 1263.41 | 1263.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 14:15:00 | 1263.75 | 1256.84 | 1259.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-08-19 14:15:00 | 1250.72 | 1256.92 | 1259.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 1250.72 | 1256.92 | 1259.88 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-08-20 09:15:00 | 1260.50 | 1256.91 | 1259.84 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 09:15:00 | 1280.75 | 1260.98 | 1260.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 11:15:00 | 1289.22 | 1261.50 | 1261.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1307.25 | 1308.17 | 1289.22 | EMA200 retest candle locked |

### Cycle 6 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 1171.00 | 1277.63 | 1277.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 1167.45 | 1256.52 | 1266.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 1113.50 | 1106.49 | 1137.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 1101.50 | 1106.86 | 1137.02 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1128.22 | 1106.31 | 1133.56 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-10 10:15:00 | 1134.55 | 1107.91 | 1133.44 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 09:15:00 | 1151.40 | 1115.71 | 1115.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 1169.47 | 1116.24 | 1115.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1165.00 | 1165.30 | 1145.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 10:15:00 | 1173.70 | 1165.55 | 1146.59 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 1149.05 | 1166.50 | 1149.51 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 1115.85 | 1187.84 | 1187.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 1111.75 | 1169.13 | 1177.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1161.60 | 1146.26 | 1163.57 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 1206.00 | 1171.03 | 1170.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1216.10 | 1171.86 | 1171.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 14:15:00 | 1186.60 | 1187.17 | 1180.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-07 12:15:00 | 1190.70 | 1177.63 | 1176.77 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-10-07 14:15:00 | 1176.20 | 1177.69 | 1176.81 | Close below EMA400 |

### Cycle 10 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 1233.70 | 1278.41 | 1278.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1221.20 | 1277.40 | 1277.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 15:15:00 | 1222.80 | 1222.46 | 1243.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 11:15:00 | 1210.10 | 1222.40 | 1242.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 13:15:00 | 1240.90 | 1222.73 | 1241.32 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 14:15:00 | 1248.70 | 1222.98 | 1241.36 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 1410.50 | 1255.11 | 1254.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 1412.80 | 1258.18 | 1256.01 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-04 09:15:00 | 1133.18 | 2023-10-19 12:15:00 | 1181.07 | TARGET | 47.90 |
| SELL | 2024-05-03 11:15:00 | 1233.57 | 2024-05-08 11:15:00 | 1262.20 | EXIT_EMA400 | -28.62 |
| SELL | 2024-05-07 13:15:00 | 1252.60 | 2024-05-08 11:15:00 | 1262.20 | EXIT_EMA400 | -9.60 |
| SELL | 2024-08-19 14:15:00 | 1250.72 | 2024-08-20 09:15:00 | 1260.50 | EXIT_EMA400 | -9.78 |
| SELL | 2025-01-06 09:15:00 | 1101.50 | 2025-01-10 10:15:00 | 1134.55 | EXIT_EMA400 | -33.05 |
| BUY | 2025-05-05 10:15:00 | 1173.70 | 2025-05-09 09:15:00 | 1149.05 | EXIT_EMA400 | -24.65 |
| BUY | 2025-10-07 12:15:00 | 1190.70 | 2025-10-07 14:15:00 | 1176.20 | EXIT_EMA400 | -14.50 |
| SELL | 2026-04-08 11:15:00 | 1210.10 | 2026-04-10 14:15:00 | 1248.70 | EXIT_EMA400 | -38.60 |
