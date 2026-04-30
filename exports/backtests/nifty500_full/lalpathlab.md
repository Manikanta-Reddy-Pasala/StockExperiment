# Dr. Lal Path Labs Ltd. (LALPATHLAB.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1367.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 7 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -194.37
- **Avg P&L per closed trade:** -24.30

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 14:15:00 | 1226.70 | 1270.08 | 1270.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 09:15:00 | 1201.35 | 1265.95 | 1268.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-30 13:15:00 | 1253.28 | 1251.68 | 1260.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-01-30 14:15:00 | 1235.50 | 1251.52 | 1259.88 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 1255.00 | 1251.41 | 1259.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-01-31 12:15:00 | 1260.03 | 1251.42 | 1259.63 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 1249.00 | 1164.25 | 1164.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1263.25 | 1168.92 | 1166.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 13:15:00 | 1649.97 | 1654.14 | 1580.58 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-09-24 09:15:00 | 1660.35 | 1649.34 | 1588.50 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 11:15:00 | 1650.62 | 1690.22 | 1640.14 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 09:15:00 | 1682.97 | 1688.70 | 1640.61 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1648.40 | 1686.23 | 1642.38 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-22 09:15:00 | 1657.72 | 1685.95 | 1642.45 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1655.82 | 1685.65 | 1642.52 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-22 11:15:00 | 1666.50 | 1685.46 | 1642.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1650.97 | 1684.05 | 1642.98 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-10-23 13:15:00 | 1639.78 | 1682.61 | 1643.07 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 1530.65 | 1614.01 | 1614.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1523.00 | 1603.42 | 1608.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 14:15:00 | 1540.03 | 1535.77 | 1562.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-09 09:15:00 | 1526.80 | 1535.66 | 1562.21 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-09 12:15:00 | 1564.85 | 1536.21 | 1562.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 1391.95 | 1335.66 | 1335.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 1398.40 | 1340.71 | 1338.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 1391.20 | 1391.90 | 1372.37 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-03 10:15:00 | 1417.15 | 1392.16 | 1373.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1405.20 | 1430.29 | 1404.49 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-23 14:15:00 | 1402.00 | 1430.01 | 1404.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1543.00 | 1577.32 | 1577.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1536.80 | 1572.99 | 1575.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1616.00 | 1572.91 | 1574.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-04 10:15:00 | 1573.90 | 1576.30 | 1576.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1573.90 | 1576.30 | 1576.65 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-11-04 11:15:00 | 1582.10 | 1576.36 | 1576.68 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-01-30 14:15:00 | 1235.50 | 2024-01-31 12:15:00 | 1260.03 | EXIT_EMA400 | -24.53 |
| BUY | 2024-09-24 09:15:00 | 1660.35 | 2024-10-23 13:15:00 | 1639.78 | EXIT_EMA400 | -20.57 |
| BUY | 2024-10-18 09:15:00 | 1682.97 | 2024-10-23 13:15:00 | 1639.78 | EXIT_EMA400 | -43.20 |
| BUY | 2024-10-22 09:15:00 | 1657.72 | 2024-10-23 13:15:00 | 1639.78 | EXIT_EMA400 | -17.95 |
| BUY | 2024-10-22 11:15:00 | 1666.50 | 2024-10-23 13:15:00 | 1639.78 | EXIT_EMA400 | -26.72 |
| SELL | 2024-12-09 09:15:00 | 1526.80 | 2024-12-09 12:15:00 | 1564.85 | EXIT_EMA400 | -38.05 |
| BUY | 2025-06-03 10:15:00 | 1417.15 | 2025-06-23 14:15:00 | 1402.00 | EXIT_EMA400 | -15.15 |
| SELL | 2025-11-04 10:15:00 | 1573.90 | 2025-11-04 11:15:00 | 1582.10 | EXIT_EMA400 | -8.20 |
