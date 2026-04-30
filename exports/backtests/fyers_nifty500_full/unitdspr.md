# United Spirits Ltd. (UNITDSPR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1328.00
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
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -163.05
- **Avg P&L per closed trade:** -23.29

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 1452.20 | 1517.82 | 1517.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 1451.80 | 1517.17 | 1517.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-01 12:15:00 | 1486.30 | 1480.26 | 1496.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 1393.05 | 1467.52 | 1486.66 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 1396.95 | 1360.67 | 1398.87 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-24 12:15:00 | 1387.35 | 1363.74 | 1398.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1386.40 | 1364.83 | 1398.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-25 10:15:00 | 1378.45 | 1364.97 | 1398.34 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 1390.45 | 1365.90 | 1396.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-27 13:15:00 | 1396.70 | 1366.21 | 1396.25 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 12:15:00 | 1521.00 | 1413.16 | 1412.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 1536.30 | 1424.11 | 1418.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1529.10 | 1531.58 | 1497.23 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-02 13:15:00 | 1540.60 | 1529.44 | 1500.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-11 09:15:00 | 1503.80 | 1553.66 | 1519.91 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 1452.40 | 1499.28 | 1499.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 1441.60 | 1497.33 | 1498.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 1334.00 | 1328.83 | 1364.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 14:15:00 | 1320.40 | 1329.76 | 1361.96 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-22 09:15:00 | 1351.70 | 1323.92 | 1347.63 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 10:15:00 | 1459.90 | 1348.20 | 1348.14 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 11:15:00 | 1318.50 | 1401.31 | 1401.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 1281.10 | 1377.35 | 1388.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 1360.90 | 1360.18 | 1376.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-01 09:15:00 | 1338.50 | 1360.00 | 1376.03 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-04 09:15:00 | 1374.10 | 1357.51 | 1373.08 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 11:15:00 | 1406.00 | 1381.97 | 1381.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 13:15:00 | 1418.80 | 1383.81 | 1382.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 10:15:00 | 1386.40 | 1388.48 | 1385.45 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 14:15:00 | 1314.00 | 1382.78 | 1382.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 1312.90 | 1379.15 | 1381.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 1392.20 | 1376.77 | 1379.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-17 14:15:00 | 1300.90 | 1366.05 | 1373.46 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-21 09:15:00 | 1329.00 | 1289.81 | 1318.96 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 09:15:00 | 1393.05 | 2025-03-27 13:15:00 | 1396.70 | EXIT_EMA400 | -3.65 |
| SELL | 2025-03-24 12:15:00 | 1387.35 | 2025-03-27 13:15:00 | 1396.70 | EXIT_EMA400 | -9.35 |
| SELL | 2025-03-25 10:15:00 | 1378.45 | 2025-03-27 13:15:00 | 1396.70 | EXIT_EMA400 | -18.25 |
| BUY | 2025-06-02 13:15:00 | 1540.60 | 2025-06-11 09:15:00 | 1503.80 | EXIT_EMA400 | -36.80 |
| SELL | 2025-09-04 14:15:00 | 1320.40 | 2025-09-22 09:15:00 | 1351.70 | EXIT_EMA400 | -31.30 |
| SELL | 2026-02-01 09:15:00 | 1338.50 | 2026-02-04 09:15:00 | 1374.10 | EXIT_EMA400 | -35.60 |
| SELL | 2026-03-17 14:15:00 | 1300.90 | 2026-04-21 09:15:00 | 1329.00 | EXIT_EMA400 | -28.10 |
