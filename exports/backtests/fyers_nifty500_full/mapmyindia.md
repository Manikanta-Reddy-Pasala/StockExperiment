# C.E. Info Systems Ltd. (MAPMYINDIA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 923.50
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 0 |
| EXIT | 5 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 1 / 4
- **Target hits / EMA400 exits:** 1 / 4
- **Total realized P&L (per unit):** -189.23
- **Avg P&L per closed trade:** -37.85

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 15:15:00 | 2023.90 | 2177.72 | 2178.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 1993.00 | 2156.27 | 2167.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2118.55 | 2111.08 | 2137.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-24 14:15:00 | 2065.55 | 2108.56 | 2134.39 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-27 09:15:00 | 2145.90 | 2103.99 | 2130.01 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 13:15:00 | 1732.00 | 1677.14 | 1677.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 1750.70 | 1677.87 | 1677.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 11:15:00 | 1910.00 | 1913.13 | 1844.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 11:15:00 | 1921.40 | 1913.06 | 1846.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-12 09:15:00 | 1796.00 | 1914.61 | 1859.40 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1744.80 | 1822.33 | 1822.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1807.43 | 1814.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1798.00 | 1793.76 | 1805.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-15 12:15:00 | 1779.00 | 1793.35 | 1804.63 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-15 14:15:00 | 1807.00 | 1793.38 | 1804.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1818.10 | 1722.19 | 1721.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1854.00 | 1729.18 | 1725.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 15:15:00 | 1760.40 | 1763.45 | 1747.02 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-07 11:15:00 | 1768.60 | 1763.31 | 1747.19 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-11 09:15:00 | 1734.00 | 1767.58 | 1750.34 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1700.20 | 1737.93 | 1738.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1679.00 | 1735.89 | 1737.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-04 09:15:00 | 1685.30 | 1712.99 | 1723.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-22 10:15:00 | 1705.00 | 1676.85 | 1698.11 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-24 14:15:00 | 2065.55 | 2024-09-27 09:15:00 | 2145.90 | EXIT_EMA400 | -80.35 |
| BUY | 2025-06-04 11:15:00 | 1921.40 | 2025-06-12 09:15:00 | 1796.00 | EXIT_EMA400 | -125.40 |
| SELL | 2025-07-15 12:15:00 | 1779.00 | 2025-07-15 14:15:00 | 1807.00 | EXIT_EMA400 | -28.00 |
| BUY | 2025-11-07 11:15:00 | 1768.60 | 2025-11-10 14:15:00 | 1832.82 | TARGET | 64.22 |
| SELL | 2025-12-04 09:15:00 | 1685.30 | 2025-12-22 10:15:00 | 1705.00 | EXIT_EMA400 | -19.70 |
