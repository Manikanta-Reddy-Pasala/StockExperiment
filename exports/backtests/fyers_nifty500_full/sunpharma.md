# Sun Pharmaceutical Industries Ltd. (SUNPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1813.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 4 |
| ENTRY1 | 5 |
| ENTRY2 | 3 |
| EXIT | 5 |

## P&L

- **Trades closed:** 8
- **Trades open at end:** 0
- **Winners / losers:** 0 / 8
- **Target hits / EMA400 exits:** 0 / 8
- **Total realized P&L (per unit):** -277.85
- **Avg P&L per closed trade:** -34.73

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 1745.50 | 1811.76 | 1811.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1729.80 | 1810.94 | 1811.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.64 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-03 09:15:00 | 1790.60 | 1805.44 | 1808.44 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1790.60 | 1805.44 | 1808.44 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-05 10:15:00 | 1782.50 | 1804.23 | 1807.60 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1806.85 | 1804.26 | 1807.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-05 13:15:00 | 1815.35 | 1804.37 | 1807.62 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.03 | 1808.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.35 | 1809.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1832.80 | 1834.09 | 1823.70 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1748.65 | 1815.25 | 1815.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1717.30 | 1806.28 | 1810.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1668.80 | 1668.44 | 1713.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-07 09:15:00 | 1649.50 | 1711.56 | 1721.97 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 1695.70 | 1704.52 | 1717.23 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-04-11 10:15:00 | 1689.15 | 1704.37 | 1717.09 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 1718.20 | 1703.85 | 1716.44 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 1781.80 | 1725.07 | 1724.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1817.90 | 1726.00 | 1725.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 1755.00 | 1762.28 | 1746.45 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1681.10 | 1736.02 | 1736.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1670.40 | 1735.37 | 1735.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-12 13:15:00 | 1691.50 | 1702.86 | 1715.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-30 10:15:00 | 1700.80 | 1683.04 | 1699.21 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1691.90 | 1640.23 | 1640.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1701.40 | 1643.22 | 1641.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 1754.20 | 1771.79 | 1737.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-07 09:15:00 | 1798.00 | 1749.58 | 1736.57 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-09 12:15:00 | 1738.60 | 1751.91 | 1738.86 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1587.80 | 1697.02 | 1711.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1724.48 | 1716.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.04 | 1739.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-20 11:15:00 | 1772.20 | 1759.96 | 1740.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 1747.10 | 1760.70 | 1741.68 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-03-24 09:15:00 | 1761.80 | 1760.71 | 1741.78 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.60 | 1747.15 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1711.32 | 1721.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-03 09:15:00 | 1790.60 | 2024-12-05 13:15:00 | 1815.35 | EXIT_EMA400 | -24.75 |
| SELL | 2024-12-05 10:15:00 | 1782.50 | 2024-12-05 13:15:00 | 1815.35 | EXIT_EMA400 | -32.85 |
| SELL | 2025-04-07 09:15:00 | 1649.50 | 2025-04-15 09:15:00 | 1718.20 | EXIT_EMA400 | -68.70 |
| SELL | 2025-04-11 10:15:00 | 1689.15 | 2025-04-15 09:15:00 | 1718.20 | EXIT_EMA400 | -29.05 |
| SELL | 2025-06-12 13:15:00 | 1691.50 | 2025-06-30 10:15:00 | 1700.80 | EXIT_EMA400 | -9.30 |
| BUY | 2026-01-07 09:15:00 | 1798.00 | 2026-01-09 12:15:00 | 1738.60 | EXIT_EMA400 | -59.40 |
| BUY | 2026-03-20 11:15:00 | 1772.20 | 2026-04-01 12:15:00 | 1740.10 | EXIT_EMA400 | -32.10 |
| BUY | 2026-03-24 09:15:00 | 1761.80 | 2026-04-01 12:15:00 | 1740.10 | EXIT_EMA400 | -21.70 |
