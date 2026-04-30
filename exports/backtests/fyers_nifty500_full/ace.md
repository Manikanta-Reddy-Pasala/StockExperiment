# Action Construction Equipment Ltd. (ACE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 890.55
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -66.85
- **Avg P&L per closed trade:** -11.14

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 13:15:00 | 1288.85 | 1414.68 | 1415.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 14:15:00 | 1283.00 | 1413.37 | 1414.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 13:15:00 | 1315.40 | 1292.25 | 1327.59 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 1392.30 | 1349.93 | 1349.91 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 1296.95 | 1349.66 | 1349.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 14:15:00 | 1284.00 | 1347.28 | 1348.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1346.15 | 1342.25 | 1345.98 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-17 09:15:00 | 1413.70 | 1348.79 | 1348.69 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 1277.00 | 1349.14 | 1349.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 1273.20 | 1348.38 | 1348.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 15:15:00 | 1321.05 | 1310.88 | 1328.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-31 09:15:00 | 1291.25 | 1310.68 | 1328.14 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-10-31 14:15:00 | 1347.45 | 1310.53 | 1327.63 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 1424.00 | 1319.39 | 1319.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 1426.20 | 1320.45 | 1319.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 15:15:00 | 1332.00 | 1335.15 | 1327.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-16 09:15:00 | 1377.75 | 1335.58 | 1327.90 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 1402.60 | 1441.00 | 1399.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-09 14:15:00 | 1398.25 | 1440.57 | 1399.83 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 13:15:00 | 1248.70 | 1370.15 | 1370.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 15:15:00 | 1238.95 | 1356.39 | 1363.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 11:15:00 | 1180.25 | 1175.19 | 1234.09 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-07 09:15:00 | 1143.40 | 1212.88 | 1223.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-05-08 11:15:00 | 1231.00 | 1210.30 | 1221.73 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 1277.00 | 1227.16 | 1227.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 14:15:00 | 1294.00 | 1229.39 | 1228.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 1246.00 | 1247.94 | 1239.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-09 09:15:00 | 1264.60 | 1247.93 | 1240.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1242.90 | 1249.58 | 1241.77 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-12 13:15:00 | 1227.00 | 1249.31 | 1241.71 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1179.00 | 1234.98 | 1235.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1172.00 | 1234.35 | 1234.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 09:15:00 | 1224.10 | 1223.55 | 1229.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-25 11:15:00 | 1212.40 | 1223.38 | 1228.87 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1225.70 | 1222.94 | 1228.32 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-27 14:15:00 | 1212.00 | 1222.76 | 1228.10 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-30 09:15:00 | 1229.70 | 1222.74 | 1228.04 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 916.55 | 879.60 | 879.57 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-31 09:15:00 | 1291.25 | 2024-10-31 14:15:00 | 1347.45 | EXIT_EMA400 | -56.20 |
| BUY | 2024-12-16 09:15:00 | 1377.75 | 2024-12-20 10:15:00 | 1527.30 | TARGET | 149.55 |
| SELL | 2025-05-07 09:15:00 | 1143.40 | 2025-05-08 11:15:00 | 1231.00 | EXIT_EMA400 | -87.60 |
| BUY | 2025-06-09 09:15:00 | 1264.60 | 2025-06-12 13:15:00 | 1227.00 | EXIT_EMA400 | -37.60 |
| SELL | 2025-06-25 11:15:00 | 1212.40 | 2025-06-30 09:15:00 | 1229.70 | EXIT_EMA400 | -17.30 |
| SELL | 2025-06-27 14:15:00 | 1212.00 | 2025-06-30 09:15:00 | 1229.70 | EXIT_EMA400 | -17.70 |
