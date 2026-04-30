# C.E. Info Systems Ltd. (MAPMYINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 919.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -274.04
- **Avg P&L per closed trade:** -45.67

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 15:15:00 | 1966.80 | 2023.31 | 2023.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 09:15:00 | 1948.60 | 2022.56 | 2023.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-21 11:15:00 | 1800.30 | 1792.00 | 1855.96 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-04-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 15:15:00 | 2020.00 | 1876.46 | 1876.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 12:15:00 | 2035.05 | 1897.88 | 1887.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 1919.95 | 1924.10 | 1903.25 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-14 09:15:00 | 2009.00 | 1902.75 | 1895.21 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-05-28 15:15:00 | 1910.00 | 1943.23 | 1921.65 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 2079.30 | 2193.54 | 2193.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 2066.70 | 2188.90 | 2191.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2116.00 | 2111.19 | 2142.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-24 13:15:00 | 2079.40 | 2109.07 | 2139.34 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-27 09:15:00 | 2145.90 | 2104.03 | 2134.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 14:15:00 | 1750.70 | 1678.19 | 1677.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 15:15:00 | 1765.60 | 1679.06 | 1678.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 1910.00 | 1913.28 | 1844.39 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 11:15:00 | 1921.40 | 1913.20 | 1846.72 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 09:15:00 | 1796.70 | 1914.72 | 1859.59 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1744.80 | 1822.38 | 1822.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1807.48 | 1814.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1798.00 | 1793.83 | 1805.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-15 12:15:00 | 1779.00 | 1793.39 | 1804.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 14:15:00 | 1807.00 | 1793.42 | 1804.63 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1818.10 | 1722.14 | 1721.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1854.00 | 1729.13 | 1725.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1752.10 | 1763.51 | 1747.17 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-07 11:15:00 | 1768.60 | 1763.47 | 1747.31 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 1732.60 | 1767.71 | 1750.44 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1700.20 | 1738.11 | 1738.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1679.00 | 1736.06 | 1737.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1729.50 | 1713.24 | 1724.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 1685.30 | 1713.23 | 1723.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-22 10:15:00 | 1705.00 | 1676.98 | 1698.24 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-05-14 09:15:00 | 2009.00 | 2024-05-28 15:15:00 | 1910.00 | EXIT_EMA400 | -99.00 |
| SELL | 2024-09-24 13:15:00 | 2079.40 | 2024-09-27 09:15:00 | 2145.90 | EXIT_EMA400 | -66.50 |
| BUY | 2025-06-04 11:15:00 | 1921.40 | 2025-06-12 09:15:00 | 1796.70 | EXIT_EMA400 | -124.70 |
| SELL | 2025-07-15 12:15:00 | 1779.00 | 2025-07-15 14:15:00 | 1807.00 | EXIT_EMA400 | -28.00 |
| BUY | 2025-11-07 11:15:00 | 1768.60 | 2025-11-10 14:15:00 | 1832.46 | TARGET | 63.86 |
| SELL | 2025-12-04 09:15:00 | 1685.30 | 2025-12-22 10:15:00 | 1705.00 | EXIT_EMA400 | -19.70 |
