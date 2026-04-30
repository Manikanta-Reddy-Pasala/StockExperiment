# PB Fintech Ltd. (POLICYBZR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1664.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 2 / 3
- **Target hits / EMA400 exits:** 2 / 3
- **Total realized P&L (per unit):** 70.64
- **Avg P&L per closed trade:** 14.13

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 1696.60 | 1875.64 | 1876.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 1679.55 | 1863.33 | 1870.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 1535.25 | 1506.04 | 1610.94 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 11:15:00 | 1501.80 | 1549.41 | 1603.57 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1587.60 | 1535.03 | 1587.40 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 15:15:00 | 1708.00 | 1613.49 | 1613.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1745.90 | 1614.81 | 1613.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 1638.60 | 1645.95 | 1630.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-21 09:15:00 | 1698.30 | 1646.60 | 1631.63 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 11:15:00 | 1773.40 | 1824.58 | 1774.43 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1753.50 | 1807.99 | 1808.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1722.00 | 1807.13 | 1807.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 1778.20 | 1769.41 | 1786.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-14 13:15:00 | 1705.00 | 1760.92 | 1779.10 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1748.20 | 1726.93 | 1755.64 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 1765.80 | 1727.54 | 1755.67 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 12:15:00 | 1847.80 | 1770.63 | 1770.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 1856.00 | 1771.48 | 1770.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1775.60 | 1776.61 | 1773.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 10:15:00 | 1799.40 | 1776.34 | 1773.53 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1841.30 | 1845.94 | 1816.00 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-17 09:15:00 | 1786.80 | 1844.71 | 1815.98 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1691.40 | 1812.38 | 1812.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1665.60 | 1810.92 | 1812.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1509.70 | 1505.65 | 1574.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 13:15:00 | 1468.20 | 1506.47 | 1570.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 13:15:00 | 1528.60 | 1482.52 | 1527.31 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 1703.30 | 1560.09 | 1559.43 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 11:15:00 | 1501.80 | 2025-04-15 09:15:00 | 1587.60 | EXIT_EMA400 | -85.80 |
| BUY | 2025-05-21 09:15:00 | 1698.30 | 2025-06-06 09:15:00 | 1898.32 | TARGET | 200.02 |
| SELL | 2025-10-14 13:15:00 | 1705.00 | 2025-10-28 09:15:00 | 1765.80 | EXIT_EMA400 | -60.80 |
| BUY | 2025-11-26 10:15:00 | 1799.40 | 2025-12-02 09:15:00 | 1877.02 | TARGET | 77.62 |
| SELL | 2026-03-19 13:15:00 | 1468.20 | 2026-04-16 13:15:00 | 1528.60 | EXIT_EMA400 | -60.40 |
