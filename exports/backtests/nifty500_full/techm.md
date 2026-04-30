# Tech Mahindra Ltd. (TECHM.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1473.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT3 | 4 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 5 / 4
- **Target hits / EMA400 exits:** 5 / 4
- **Total realized P&L (per unit):** 293.22
- **Avg P&L per closed trade:** 32.58

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 09:15:00 | 1106.40 | 1208.83 | 1209.04 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 15:15:00 | 1223.90 | 1191.00 | 1190.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-06 09:15:00 | 1227.90 | 1193.02 | 1191.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 11:15:00 | 1199.00 | 1203.45 | 1197.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-13 14:15:00 | 1216.20 | 1203.61 | 1198.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-01-08 12:15:00 | 1227.60 | 1247.67 | 1229.50 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 11:15:00 | 1255.25 | 1282.56 | 1282.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 1247.00 | 1281.34 | 1282.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 13:15:00 | 1274.00 | 1272.74 | 1277.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-09 11:15:00 | 1256.80 | 1271.62 | 1276.29 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-26 09:15:00 | 1312.30 | 1239.08 | 1256.12 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 10:15:00 | 1314.00 | 1264.99 | 1264.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 11:15:00 | 1315.90 | 1265.49 | 1265.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1276.00 | 1286.37 | 1277.10 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-06 09:15:00 | 1294.05 | 1275.82 | 1272.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-31 09:15:00 | 1615.40 | 1666.05 | 1622.76 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 1651.60 | 1690.74 | 1690.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1633.45 | 1687.53 | 1689.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 1682.70 | 1680.20 | 1684.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 12:15:00 | 1664.60 | 1681.12 | 1685.09 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 12:15:00 | 1664.60 | 1681.12 | 1685.09 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-02-12 11:15:00 | 1685.20 | 1680.87 | 1684.84 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.50 | 1503.54 | 1503.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.14 | 1504.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.80 | 1540.67 | 1525.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 13:15:00 | 1565.00 | 1541.83 | 1526.67 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-10 09:15:00 | 1599.80 | 1639.80 | 1604.60 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1460.30 | 1585.46 | 1585.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1447.50 | 1581.53 | 1583.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.49 | 1544.79 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-14 11:15:00 | 1502.90 | 1520.31 | 1544.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.89 | 1537.07 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 13:15:00 | 1512.30 | 1515.82 | 1536.36 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1528.50 | 1505.42 | 1524.69 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1545.90 | 1472.95 | 1472.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.68 | 1474.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.95 | 1546.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-09 09:15:00 | 1588.60 | 1580.02 | 1547.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1660.54 | 1609.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-02-04 10:15:00 | 1606.40 | 1660.00 | 1609.55 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 14:15:00 | 1439.60 | 1586.36 | 1586.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 09:15:00 | 1394.00 | 1583.02 | 1584.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.90 | 1415.56 | 1471.97 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-25 15:15:00 | 1404.00 | 1416.27 | 1469.32 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.44 | 1460.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-07 09:15:00 | 1464.00 | 1416.14 | 1460.32 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-12-13 14:15:00 | 1216.20 | 2023-12-15 09:15:00 | 1270.81 | TARGET | 54.61 |
| SELL | 2024-04-09 11:15:00 | 1256.80 | 2024-04-16 12:15:00 | 1198.34 | TARGET | 58.46 |
| BUY | 2024-06-06 09:15:00 | 1294.05 | 2024-06-07 09:15:00 | 1357.77 | TARGET | 63.72 |
| SELL | 2025-02-11 12:15:00 | 1664.60 | 2025-02-12 11:15:00 | 1685.20 | EXIT_EMA400 | -20.60 |
| BUY | 2025-06-04 13:15:00 | 1565.00 | 2025-06-16 11:15:00 | 1680.00 | TARGET | 115.00 |
| SELL | 2025-08-14 11:15:00 | 1502.90 | 2025-09-10 09:15:00 | 1528.50 | EXIT_EMA400 | -25.60 |
| SELL | 2025-08-26 13:15:00 | 1512.30 | 2025-09-10 09:15:00 | 1528.50 | EXIT_EMA400 | -16.20 |
| BUY | 2026-01-09 09:15:00 | 1588.60 | 2026-01-19 09:15:00 | 1712.43 | TARGET | 123.83 |
| SELL | 2026-03-25 15:15:00 | 1404.00 | 2026-04-07 09:15:00 | 1464.00 | EXIT_EMA400 | -60.00 |
