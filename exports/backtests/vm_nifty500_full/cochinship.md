# Cochin Shipyard Ltd. (COCHINSHIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 1733.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 3 |
| ENTRY1 | 4 |
| ENTRY2 | 3 |
| EXIT | 4 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 32.24
- **Avg P&L per closed trade:** 4.61

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 13:15:00 | 1916.25 | 2147.00 | 2147.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 14:15:00 | 1912.85 | 2144.67 | 2146.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 1504.05 | 1455.11 | 1595.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 10:15:00 | 1449.00 | 1565.98 | 1604.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 15:15:00 | 1595.00 | 1545.59 | 1587.14 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 1489.70 | 1404.98 | 1404.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1498.60 | 1411.53 | 1408.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 1413.60 | 1415.82 | 1410.57 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 09:15:00 | 1500.50 | 1416.89 | 1411.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1500.50 | 1416.89 | 1411.26 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-29 09:15:00 | 1531.30 | 1423.13 | 1414.63 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-08 15:15:00 | 1438.00 | 1462.70 | 1439.44 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1734.00 | 1883.35 | 1884.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1730.90 | 1878.91 | 1881.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 1749.00 | 1735.61 | 1788.40 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 09:15:00 | 1707.10 | 1736.15 | 1785.86 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 1780.00 | 1710.28 | 1760.32 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 10:15:00 | 1891.80 | 1795.51 | 1795.26 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1768.10 | 1800.02 | 1800.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1719.00 | 1798.94 | 1799.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1796.70 | 1784.76 | 1791.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 09:15:00 | 1728.00 | 1784.59 | 1791.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1677.30 | 1639.34 | 1686.31 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-26 14:15:00 | 1650.10 | 1640.49 | 1685.73 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1667.00 | 1640.87 | 1685.47 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-29 11:15:00 | 1648.60 | 1641.16 | 1685.17 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-28 14:15:00 | 1620.90 | 1564.98 | 1616.20 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 1688.90 | 1482.31 | 1481.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 11:15:00 | 1704.20 | 1484.51 | 1483.02 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-23 10:15:00 | 1449.00 | 2024-12-30 15:15:00 | 1595.00 | EXIT_EMA400 | -146.00 |
| BUY | 2025-04-28 09:15:00 | 1500.50 | 2025-05-08 15:15:00 | 1438.00 | EXIT_EMA400 | -62.50 |
| BUY | 2025-04-29 09:15:00 | 1531.30 | 2025-05-08 15:15:00 | 1438.00 | EXIT_EMA400 | -93.30 |
| SELL | 2025-09-04 09:15:00 | 1707.10 | 2025-09-15 09:15:00 | 1780.00 | EXIT_EMA400 | -72.90 |
| SELL | 2025-11-13 09:15:00 | 1728.00 | 2025-12-17 09:15:00 | 1537.67 | TARGET | 190.33 |
| SELL | 2025-12-26 14:15:00 | 1650.10 | 2026-01-12 09:15:00 | 1543.20 | TARGET | 106.90 |
| SELL | 2025-12-29 11:15:00 | 1648.60 | 2026-01-12 09:15:00 | 1538.89 | TARGET | 109.71 |
