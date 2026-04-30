# PB Fintech Ltd. (POLICYBZR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1666.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 7 |
| ENTRY2 | 1 |
| EXIT | 7 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 3 / 5
- **Target hits / EMA400 exits:** 3 / 5
- **Total realized P&L (per unit):** 30.97
- **Avg P&L per closed trade:** 3.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-27 11:15:00 | 683.40 | 736.29 | 736.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-27 12:15:00 | 679.25 | 735.72 | 736.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 724.40 | 723.49 | 729.39 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-06 13:15:00 | 708.75 | 723.50 | 729.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 13:15:00 | 708.75 | 723.50 | 729.28 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2023-11-07 09:15:00 | 738.00 | 723.56 | 729.22 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 09:15:00 | 789.80 | 733.46 | 733.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 14:15:00 | 802.80 | 736.34 | 734.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 785.00 | 795.02 | 772.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-14 09:15:00 | 820.50 | 795.45 | 774.05 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-20 14:15:00 | 757.75 | 797.37 | 778.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 1696.60 | 1875.86 | 1876.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 1679.55 | 1863.54 | 1870.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 10:15:00 | 1535.20 | 1506.94 | 1612.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 11:15:00 | 1501.80 | 1549.95 | 1604.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 1583.60 | 1535.65 | 1588.22 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-15 11:15:00 | 1596.70 | 1536.26 | 1588.27 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 09:15:00 | 1745.90 | 1615.05 | 1614.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 11:15:00 | 1750.20 | 1617.72 | 1615.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 1638.00 | 1646.13 | 1631.41 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-21 09:15:00 | 1698.30 | 1646.77 | 1632.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-04 11:15:00 | 1773.40 | 1824.54 | 1774.52 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1753.50 | 1807.89 | 1808.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1722.00 | 1807.04 | 1807.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 1778.00 | 1769.21 | 1786.48 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-14 12:15:00 | 1708.80 | 1761.38 | 1779.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1748.10 | 1726.90 | 1755.57 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-28 09:15:00 | 1765.80 | 1727.51 | 1755.59 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 12:15:00 | 1847.80 | 1770.53 | 1770.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 1856.00 | 1771.38 | 1770.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 09:15:00 | 1775.60 | 1776.58 | 1773.49 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-26 09:15:00 | 1791.50 | 1776.06 | 1773.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1791.50 | 1776.06 | 1773.33 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-11-27 09:15:00 | 1801.20 | 1777.09 | 1773.95 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 1841.30 | 1845.95 | 1815.97 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-17 09:15:00 | 1785.90 | 1844.70 | 1815.94 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1691.40 | 1812.46 | 1812.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 10:15:00 | 1665.60 | 1811.00 | 1812.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1510.10 | 1506.44 | 1575.70 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-19 13:15:00 | 1468.20 | 1507.28 | 1572.43 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-16 13:15:00 | 1528.20 | 1482.80 | 1528.16 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 1683.30 | 1561.21 | 1560.71 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-11-06 13:15:00 | 708.75 | 2023-11-07 09:15:00 | 738.00 | EXIT_EMA400 | -29.25 |
| BUY | 2023-12-14 09:15:00 | 820.50 | 2023-12-20 14:15:00 | 757.75 | EXIT_EMA400 | -62.75 |
| SELL | 2025-04-04 11:15:00 | 1501.80 | 2025-04-15 11:15:00 | 1596.70 | EXIT_EMA400 | -94.90 |
| BUY | 2025-05-21 09:15:00 | 1698.30 | 2025-06-06 09:15:00 | 1896.91 | TARGET | 198.61 |
| SELL | 2025-10-14 12:15:00 | 1708.80 | 2025-10-28 09:15:00 | 1765.80 | EXIT_EMA400 | -57.00 |
| BUY | 2025-11-26 09:15:00 | 1791.50 | 2025-12-01 10:15:00 | 1846.01 | TARGET | 54.51 |
| BUY | 2025-11-27 09:15:00 | 1801.20 | 2025-12-04 10:15:00 | 1882.96 | TARGET | 81.76 |
| SELL | 2026-03-19 13:15:00 | 1468.20 | 2026-04-16 13:15:00 | 1528.20 | EXIT_EMA400 | -60.00 |
