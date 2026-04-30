# Oberoi Realty Ltd. (OBEROIRLTY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1669.60
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 1 / 7
- **Target hits / EMA400 exits:** 1 / 7
- **Total realized P&L (per unit):** -170.01
- **Avg P&L per closed trade:** -21.25

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 12:15:00 | 1305.00 | 1363.48 | 1363.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 1298.00 | 1359.29 | 1361.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 10:15:00 | 1367.20 | 1354.02 | 1358.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-28 12:15:00 | 1332.15 | 1359.08 | 1360.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 1351.60 | 1357.01 | 1359.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-01 12:15:00 | 1361.85 | 1357.00 | 1359.39 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 10:15:00 | 1383.95 | 1361.53 | 1361.49 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 10:15:00 | 1336.25 | 1361.33 | 1361.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 11:15:00 | 1324.05 | 1360.96 | 1361.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-19 10:15:00 | 1352.85 | 1350.59 | 1355.49 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-03-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 12:15:00 | 1447.20 | 1359.67 | 1359.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 13:15:00 | 1455.60 | 1360.62 | 1360.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-16 09:15:00 | 1455.40 | 1456.46 | 1419.29 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-16 10:15:00 | 1464.75 | 1456.54 | 1419.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-04-18 13:15:00 | 1420.40 | 1455.87 | 1420.99 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1789.70 | 2061.33 | 2061.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1766.75 | 2058.40 | 2059.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 1624.10 | 1620.81 | 1731.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 09:15:00 | 1614.65 | 1628.48 | 1718.53 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 1699.00 | 1603.23 | 1667.14 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1740.70 | 1668.21 | 1667.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 1757.80 | 1672.97 | 1670.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1865.00 | 1865.91 | 1804.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 14:15:00 | 1872.00 | 1866.01 | 1805.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-14 10:15:00 | 1805.00 | 1857.39 | 1810.93 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1618.30 | 1789.61 | 1789.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1605.60 | 1776.84 | 1783.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 1669.80 | 1668.41 | 1706.28 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-02 13:15:00 | 1652.40 | 1668.19 | 1705.61 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1675.40 | 1647.68 | 1680.93 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-18 11:15:00 | 1668.40 | 1647.89 | 1680.87 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1675.00 | 1648.88 | 1679.92 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-22 10:15:00 | 1686.00 | 1649.97 | 1679.86 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1781.40 | 1666.64 | 1666.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 1784.00 | 1667.81 | 1666.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1710.60 | 1719.07 | 1698.01 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1632.60 | 1685.15 | 1685.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1623.70 | 1678.57 | 1681.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1661.80 | 1660.33 | 1670.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-30 12:15:00 | 1640.60 | 1664.16 | 1670.47 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1663.60 | 1663.55 | 1670.01 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-31 12:15:00 | 1671.80 | 1663.68 | 1670.01 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 11:15:00 | 1733.00 | 1675.86 | 1675.58 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1645.00 | 1676.15 | 1676.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1568.80 | 1673.37 | 1674.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 13:15:00 | 1576.10 | 1569.95 | 1606.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-12 09:15:00 | 1560.50 | 1570.02 | 1606.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 09:15:00 | 1519.30 | 1478.12 | 1516.68 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 11:15:00 | 1703.30 | 1545.98 | 1545.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1720.90 | 1554.10 | 1549.41 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-28 12:15:00 | 1332.15 | 2024-03-01 12:15:00 | 1361.85 | EXIT_EMA400 | -29.70 |
| BUY | 2024-04-16 10:15:00 | 1464.75 | 2024-04-18 13:15:00 | 1420.40 | EXIT_EMA400 | -44.35 |
| SELL | 2025-03-27 09:15:00 | 1614.65 | 2025-04-22 09:15:00 | 1699.00 | EXIT_EMA400 | -84.35 |
| BUY | 2025-07-04 14:15:00 | 1872.00 | 2025-07-14 10:15:00 | 1805.00 | EXIT_EMA400 | -67.00 |
| SELL | 2025-09-02 13:15:00 | 1652.40 | 2025-09-22 10:15:00 | 1686.00 | EXIT_EMA400 | -33.60 |
| SELL | 2025-09-18 11:15:00 | 1668.40 | 2025-09-22 10:15:00 | 1686.00 | EXIT_EMA400 | -17.60 |
| SELL | 2025-12-30 12:15:00 | 1640.60 | 2025-12-31 12:15:00 | 1671.80 | EXIT_EMA400 | -31.20 |
| SELL | 2026-02-12 09:15:00 | 1560.50 | 2026-03-16 10:15:00 | 1422.71 | TARGET | 137.79 |
