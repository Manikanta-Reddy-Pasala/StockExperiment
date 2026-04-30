# Tega Industries Ltd. (TEGA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1677.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 78.73
- **Avg P&L per closed trade:** 19.68

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 13:15:00 | 1707.40 | 1845.83 | 1846.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 15:15:00 | 1705.00 | 1843.07 | 1845.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 1627.65 | 1622.24 | 1686.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 1591.10 | 1628.08 | 1678.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 11:15:00 | 1629.60 | 1566.18 | 1625.32 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1622.00 | 1427.11 | 1426.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1633.40 | 1440.20 | 1433.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 1512.10 | 1519.08 | 1483.63 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 09:15:00 | 1556.80 | 1503.56 | 1485.52 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 1882.00 | 1959.24 | 1886.15 | Close below EMA400 |

### Cycle 3 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 1880.30 | 1917.33 | 1917.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1863.00 | 1912.87 | 1915.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1835.60 | 1808.11 | 1851.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 1780.00 | 1808.76 | 1849.65 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1832.90 | 1806.97 | 1845.95 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-11 09:15:00 | 1778.80 | 1807.09 | 1844.67 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1776.00 | 1745.99 | 1798.03 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 1827.70 | 1746.81 | 1798.17 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-22 09:15:00 | 1591.10 | 2025-02-05 11:15:00 | 1629.60 | EXIT_EMA400 | -38.50 |
| BUY | 2025-06-30 09:15:00 | 1556.80 | 2025-07-07 09:15:00 | 1770.63 | TARGET | 213.83 |
| SELL | 2026-02-06 09:15:00 | 1780.00 | 2026-02-25 09:15:00 | 1827.70 | EXIT_EMA400 | -47.70 |
| SELL | 2026-02-11 09:15:00 | 1778.80 | 2026-02-25 09:15:00 | 1827.70 | EXIT_EMA400 | -48.90 |
