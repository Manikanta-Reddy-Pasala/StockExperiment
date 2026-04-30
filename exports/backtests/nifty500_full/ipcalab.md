# Ipca Laboratories Ltd. (IPCALAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1530.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 0 / 7
- **Target hits / EMA400 exits:** 0 / 7
- **Total realized P&L (per unit):** -314.50
- **Avg P&L per closed trade:** -44.93

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-06-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-07 15:15:00 | 1173.00 | 1249.27 | 1249.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 11:15:00 | 1164.15 | 1228.47 | 1237.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 10:15:00 | 1180.15 | 1173.10 | 1201.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 13:15:00 | 1295.80 | 1213.21 | 1212.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1299.65 | 1218.73 | 1215.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1560.00 | 1563.34 | 1502.01 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-11-12 10:15:00 | 1596.50 | 1561.88 | 1507.64 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-13 11:15:00 | 1509.40 | 1561.18 | 1509.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 13:15:00 | 1420.20 | 1562.94 | 1563.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 12:15:00 | 1412.30 | 1515.52 | 1535.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 13:15:00 | 1515.30 | 1504.48 | 1528.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-14 09:15:00 | 1454.80 | 1503.66 | 1527.94 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 1416.75 | 1393.51 | 1441.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-24 13:15:00 | 1455.40 | 1395.35 | 1441.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1482.20 | 1420.09 | 1419.95 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 1379.30 | 1419.99 | 1420.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 10:15:00 | 1376.50 | 1419.56 | 1419.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1415.10 | 1407.66 | 1413.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 13:15:00 | 1381.40 | 1407.32 | 1413.17 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1383.00 | 1389.73 | 1402.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-24 14:15:00 | 1332.30 | 1384.05 | 1398.62 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1389.80 | 1375.77 | 1392.17 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-07-01 09:15:00 | 1370.90 | 1375.72 | 1392.06 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-07-03 09:15:00 | 1412.20 | 1375.41 | 1390.79 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1450.20 | 1402.91 | 1402.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 12:15:00 | 1454.10 | 1403.42 | 1402.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1444.20 | 1454.70 | 1434.02 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1371.70 | 1420.90 | 1421.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1411.29 | 1416.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1417.80 | 1408.84 | 1414.60 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-25 11:15:00 | 1392.80 | 1408.66 | 1414.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-08-26 14:15:00 | 1433.90 | 1407.19 | 1413.31 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1441.10 | 1349.06 | 1348.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1448.00 | 1352.61 | 1350.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1399.00 | 1414.81 | 1391.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 15:15:00 | 1429.70 | 1412.86 | 1391.61 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-30 10:15:00 | 1396.00 | 1415.93 | 1397.55 | Close below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-11-12 10:15:00 | 1596.50 | 2024-11-13 11:15:00 | 1509.40 | EXIT_EMA400 | -87.10 |
| SELL | 2025-02-14 09:15:00 | 1454.80 | 2025-03-24 13:15:00 | 1455.40 | EXIT_EMA400 | -0.60 |
| SELL | 2025-06-12 13:15:00 | 1381.40 | 2025-07-03 09:15:00 | 1412.20 | EXIT_EMA400 | -30.80 |
| SELL | 2025-06-24 14:15:00 | 1332.30 | 2025-07-03 09:15:00 | 1412.20 | EXIT_EMA400 | -79.90 |
| SELL | 2025-07-01 09:15:00 | 1370.90 | 2025-07-03 09:15:00 | 1412.20 | EXIT_EMA400 | -41.30 |
| SELL | 2025-08-25 11:15:00 | 1392.80 | 2025-08-26 14:15:00 | 1433.90 | EXIT_EMA400 | -41.10 |
| BUY | 2025-12-18 15:15:00 | 1429.70 | 2025-12-30 10:15:00 | 1396.00 | EXIT_EMA400 | -33.70 |
