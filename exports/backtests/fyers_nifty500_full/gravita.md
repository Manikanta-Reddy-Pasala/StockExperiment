# Gravita India Ltd. (GRAVITA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1626.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -9.45
- **Avg P&L per closed trade:** -2.36

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 11:15:00 | 2130.00 | 2206.90 | 2207.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-23 12:15:00 | 2081.90 | 2202.28 | 2204.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 12:15:00 | 1755.00 | 1738.85 | 1861.76 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-04 12:15:00 | 1720.00 | 1762.39 | 1839.36 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 1772.95 | 1731.52 | 1812.67 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-04-11 14:15:00 | 1853.00 | 1733.23 | 1812.72 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 10:15:00 | 1976.90 | 1854.90 | 1854.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 2010.00 | 1856.44 | 1855.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 1917.10 | 1921.39 | 1893.29 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1764.30 | 1877.85 | 1877.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1753.20 | 1876.61 | 1877.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 11:15:00 | 1803.20 | 1801.32 | 1833.29 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 09:15:00 | 1780.00 | 1818.65 | 1834.69 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-24 09:15:00 | 1809.00 | 1774.14 | 1804.07 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 11:15:00 | 1754.50 | 1683.94 | 1683.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 13:15:00 | 1783.90 | 1694.75 | 1689.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1815.00 | 1817.84 | 1780.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-06 09:15:00 | 1822.90 | 1817.89 | 1780.88 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1787.00 | 1816.71 | 1782.44 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-07 15:15:00 | 1782.00 | 1816.36 | 1782.43 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1561.50 | 1754.53 | 1755.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 1547.80 | 1737.27 | 1746.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 10:15:00 | 1674.90 | 1651.42 | 1693.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-18 14:15:00 | 1618.00 | 1656.41 | 1682.48 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-09 11:15:00 | 1539.50 | 1453.80 | 1526.08 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-04-04 12:15:00 | 1720.00 | 2025-04-11 14:15:00 | 1853.00 | EXIT_EMA400 | -133.00 |
| SELL | 2025-07-10 09:15:00 | 1780.00 | 2025-07-24 09:15:00 | 1809.00 | EXIT_EMA400 | -29.00 |
| BUY | 2026-01-06 09:15:00 | 1822.90 | 2026-01-07 15:15:00 | 1782.00 | EXIT_EMA400 | -40.90 |
| SELL | 2026-02-18 14:15:00 | 1618.00 | 2026-03-13 14:15:00 | 1424.55 | TARGET | 193.45 |
