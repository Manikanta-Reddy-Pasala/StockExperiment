# Max Financial Services Ltd. (MFSL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1591.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 1
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -127.45
- **Avg P&L per closed trade:** -42.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 1141.90 | 1168.84 | 1168.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1129.00 | 1167.45 | 1168.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 14:15:00 | 1084.85 | 1083.01 | 1109.07 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-11 09:15:00 | 1066.20 | 1093.26 | 1108.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-11 09:15:00 | 1074.95 | 1048.74 | 1073.13 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 12:15:00 | 1154.95 | 1085.91 | 1085.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 10:15:00 | 1164.00 | 1099.19 | 1092.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 1558.30 | 1558.50 | 1472.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 14:15:00 | 1569.50 | 1558.61 | 1473.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-30 09:15:00 | 1496.90 | 1548.83 | 1497.79 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1550.00 | 1561.90 | 1561.93 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 09:15:00 | 1583.10 | 1562.02 | 1561.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 12:15:00 | 1593.50 | 1562.76 | 1562.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 14:15:00 | 1672.20 | 1673.54 | 1641.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-18 11:15:00 | 1686.00 | 1672.48 | 1642.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1650.80 | 1675.26 | 1650.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-29 15:15:00 | 1639.90 | 1674.90 | 1650.31 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 1597.00 | 1649.33 | 1649.41 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 1708.40 | 1649.17 | 1649.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 1721.00 | 1657.05 | 1653.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-04 09:15:00 | 1753.20 | 1771.98 | 1726.60 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2026-03-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 15:15:00 | 1560.30 | 1704.12 | 1704.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-24 10:15:00 | 1550.00 | 1701.31 | 1702.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 09:15:00 | 1627.50 | 1618.76 | 1653.92 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-23 11:15:00 | 1597.90 | 1638.01 | 1656.36 | Sell entry 1 (retest1 break) |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-11 09:15:00 | 1066.20 | 2025-03-11 09:15:00 | 1074.95 | EXIT_EMA400 | -8.75 |
| BUY | 2025-07-11 14:15:00 | 1569.50 | 2025-07-30 09:15:00 | 1496.90 | EXIT_EMA400 | -72.60 |
| BUY | 2025-12-18 11:15:00 | 1686.00 | 2025-12-29 15:15:00 | 1639.90 | EXIT_EMA400 | -46.10 |
