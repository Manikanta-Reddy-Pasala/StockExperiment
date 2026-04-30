# United Spirits Ltd. (UNITDSPR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-06-07 09:15:00 → 2026-04-30 15:15:00 (3260 bars)
- **Last close:** 1325.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 1 / 6
- **Target hits / EMA400 exits:** 1 / 6
- **Total realized P&L (per unit):** -19.16
- **Avg P&L per closed trade:** -2.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1455.95 | 1518.58 | 1518.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1438.50 | 1510.63 | 1514.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 1489.55 | 1482.67 | 1498.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-05 10:15:00 | 1447.55 | 1482.10 | 1497.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1396.95 | 1360.83 | 1399.48 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-24 12:15:00 | 1387.35 | 1363.89 | 1399.16 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1386.40 | 1364.94 | 1398.99 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 1378.45 | 1365.07 | 1398.89 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-03-27 13:15:00 | 1396.85 | 1366.33 | 1396.77 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 1516.00 | 1414.15 | 1413.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1535.50 | 1424.10 | 1418.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1529.30 | 1531.71 | 1497.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 13:15:00 | 1540.60 | 1529.41 | 1500.83 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-11 09:15:00 | 1503.10 | 1553.68 | 1520.02 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 1452.40 | 1499.37 | 1499.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 13:15:00 | 1451.10 | 1497.97 | 1498.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1333.50 | 1328.90 | 1364.81 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 13:15:00 | 1324.50 | 1329.89 | 1362.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-22 09:15:00 | 1351.80 | 1323.90 | 1347.65 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1459.90 | 1348.31 | 1348.22 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 1318.10 | 1401.34 | 1401.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1281.10 | 1377.39 | 1388.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 1361.30 | 1360.21 | 1376.41 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 1330.30 | 1359.94 | 1376.04 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1374.10 | 1358.96 | 1374.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-06 14:15:00 | 1377.60 | 1359.31 | 1373.19 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 13:15:00 | 1400.30 | 1383.10 | 1383.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 13:15:00 | 1418.80 | 1384.37 | 1383.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 1386.40 | 1388.99 | 1386.23 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 1325.00 | 1383.87 | 1383.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 1313.90 | 1383.18 | 1383.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 1392.20 | 1377.10 | 1380.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 14:15:00 | 1300.90 | 1366.30 | 1373.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 1329.00 | 1288.97 | 1319.35 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-05 10:15:00 | 1447.55 | 2025-02-28 11:15:00 | 1298.61 | TARGET | 148.94 |
| SELL | 2025-03-24 12:15:00 | 1387.35 | 2025-03-27 13:15:00 | 1396.85 | EXIT_EMA400 | -9.50 |
| SELL | 2025-03-25 10:15:00 | 1378.45 | 2025-03-27 13:15:00 | 1396.85 | EXIT_EMA400 | -18.40 |
| BUY | 2025-06-02 13:15:00 | 1540.60 | 2025-06-11 09:15:00 | 1503.10 | EXIT_EMA400 | -37.50 |
| SELL | 2025-09-04 13:15:00 | 1324.50 | 2025-09-22 09:15:00 | 1351.80 | EXIT_EMA400 | -27.30 |
| SELL | 2026-02-02 09:15:00 | 1330.30 | 2026-02-06 14:15:00 | 1377.60 | EXIT_EMA400 | -47.30 |
| SELL | 2026-03-17 14:15:00 | 1300.90 | 2026-04-21 09:15:00 | 1329.00 | EXIT_EMA400 | -28.10 |
