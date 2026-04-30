# Oberoi Realty Ltd. (OBEROIRLTY.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1668.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -77.73
- **Avg P&L per closed trade:** -12.95

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 12:15:00 | 1789.70 | 2061.29 | 2061.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 13:15:00 | 1766.75 | 2058.36 | 2060.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 1624.10 | 1619.73 | 1728.95 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-27 09:15:00 | 1614.65 | 1627.65 | 1716.71 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-22 09:15:00 | 1699.00 | 1602.84 | 1665.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 1746.30 | 1667.39 | 1667.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 14:15:00 | 1757.80 | 1672.81 | 1669.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1865.00 | 1865.82 | 1803.81 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-04 14:15:00 | 1872.00 | 1865.92 | 1804.78 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-14 10:15:00 | 1805.00 | 1857.28 | 1810.68 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1618.30 | 1789.48 | 1789.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1605.60 | 1776.72 | 1783.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 10:15:00 | 1669.80 | 1668.36 | 1706.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-02 13:15:00 | 1652.40 | 1668.14 | 1705.50 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1675.40 | 1647.75 | 1680.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-18 11:15:00 | 1668.40 | 1647.96 | 1680.85 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1675.00 | 1648.95 | 1679.91 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-09-22 10:15:00 | 1686.00 | 1650.06 | 1679.86 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 12:15:00 | 1780.50 | 1666.56 | 1666.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 13:15:00 | 1784.00 | 1667.73 | 1666.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1710.20 | 1719.00 | 1697.95 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 15:15:00 | 1633.00 | 1685.17 | 1685.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1623.20 | 1678.62 | 1681.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 12:15:00 | 1661.80 | 1660.44 | 1670.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-18 15:15:00 | 1650.40 | 1660.32 | 1670.42 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1669.10 | 1660.41 | 1670.42 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-19 14:15:00 | 1675.80 | 1660.51 | 1670.22 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 10:15:00 | 1727.30 | 1675.46 | 1675.40 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1645.00 | 1676.17 | 1676.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1568.80 | 1673.48 | 1674.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 15:15:00 | 1566.40 | 1565.32 | 1605.24 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-10 10:15:00 | 1554.70 | 1565.18 | 1604.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-06 09:15:00 | 1519.10 | 1477.70 | 1515.46 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 1709.50 | 1544.06 | 1543.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 1716.00 | 1550.51 | 1546.76 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-27 09:15:00 | 1614.65 | 2025-04-22 09:15:00 | 1699.00 | EXIT_EMA400 | -84.35 |
| BUY | 2025-07-04 14:15:00 | 1872.00 | 2025-07-14 10:15:00 | 1805.00 | EXIT_EMA400 | -67.00 |
| SELL | 2025-09-02 13:15:00 | 1652.40 | 2025-09-22 10:15:00 | 1686.00 | EXIT_EMA400 | -33.60 |
| SELL | 2025-09-18 11:15:00 | 1668.40 | 2025-09-22 10:15:00 | 1686.00 | EXIT_EMA400 | -17.60 |
| SELL | 2025-12-18 15:15:00 | 1650.40 | 2025-12-19 14:15:00 | 1675.80 | EXIT_EMA400 | -25.40 |
| SELL | 2026-02-10 10:15:00 | 1554.70 | 2026-03-16 10:15:00 | 1404.48 | TARGET | 150.22 |
