# Cochin Shipyard Ltd. (COCHINSHIP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1740.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 2 |
| EXIT | 4 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -29.62
- **Avg P&L per closed trade:** -4.94

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 12:15:00 | 1936.10 | 2165.50 | 2165.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 13:15:00 | 1919.80 | 2163.06 | 2164.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 1504.05 | 1454.77 | 1595.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 10:15:00 | 1449.00 | 1565.94 | 1604.19 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-12-30 15:15:00 | 1590.00 | 1545.59 | 1587.20 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 1489.70 | 1405.09 | 1404.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1498.60 | 1411.64 | 1408.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 1413.60 | 1415.93 | 1410.45 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-28 09:15:00 | 1500.40 | 1417.01 | 1411.15 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1500.40 | 1417.01 | 1411.15 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-29 09:15:00 | 1531.20 | 1423.25 | 1414.53 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-08 15:15:00 | 1438.00 | 1462.78 | 1439.36 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1734.00 | 1883.43 | 1884.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1731.00 | 1879.00 | 1881.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 1748.50 | 1735.68 | 1788.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-04 09:15:00 | 1707.40 | 1736.16 | 1785.90 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 1780.50 | 1710.31 | 1760.37 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 10:15:00 | 1891.80 | 1795.48 | 1795.27 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1768.10 | 1800.02 | 1800.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1719.00 | 1798.93 | 1799.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-11-13 09:15:00 | 1728.00 | 1784.54 | 1791.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1677.30 | 1639.39 | 1686.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-12-29 12:15:00 | 1635.00 | 1641.13 | 1684.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-01-28 14:15:00 | 1621.00 | 1565.04 | 1616.26 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1678.80 | 1481.89 | 1481.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1686.30 | 1483.92 | 1482.43 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-23 10:15:00 | 1449.00 | 2024-12-30 15:15:00 | 1590.00 | EXIT_EMA400 | -141.00 |
| BUY | 2025-04-28 09:15:00 | 1500.40 | 2025-05-08 15:15:00 | 1438.00 | EXIT_EMA400 | -62.40 |
| BUY | 2025-04-29 09:15:00 | 1531.20 | 2025-05-08 15:15:00 | 1438.00 | EXIT_EMA400 | -93.20 |
| SELL | 2025-09-04 09:15:00 | 1707.40 | 2025-09-15 09:15:00 | 1780.50 | EXIT_EMA400 | -73.10 |
| SELL | 2025-11-13 09:15:00 | 1728.00 | 2025-12-17 09:15:00 | 1537.74 | TARGET | 190.26 |
| SELL | 2025-12-29 12:15:00 | 1635.00 | 2026-01-20 09:15:00 | 1485.19 | TARGET | 149.81 |
