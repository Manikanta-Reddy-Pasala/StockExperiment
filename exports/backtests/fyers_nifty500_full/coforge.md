# Coforge Ltd. (COFORGE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1195.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 119.24
- **Avg P&L per closed trade:** 19.87

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1573.60 | 1713.52 | 1713.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1551.34 | 1711.90 | 1713.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 10:15:00 | 1592.20 | 1591.89 | 1641.15 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-05 13:15:00 | 1547.89 | 1591.15 | 1640.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1578.60 | 1546.11 | 1595.07 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-25 11:15:00 | 1601.60 | 1546.98 | 1595.02 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1684.50 | 1530.34 | 1530.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1696.10 | 1540.71 | 1535.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 1856.00 | 1856.70 | 1780.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 11:15:00 | 1874.50 | 1856.72 | 1781.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 09:15:00 | 1698.60 | 1855.79 | 1790.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1637.60 | 1752.48 | 1752.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 1630.40 | 1751.26 | 1751.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1719.80 | 1719.09 | 1734.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 1657.30 | 1728.66 | 1735.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-10 09:15:00 | 1754.20 | 1719.54 | 1729.81 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1810.70 | 1737.87 | 1737.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1816.00 | 1740.10 | 1738.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1636.10 | 1737.68 | 1738.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1629.70 | 1736.61 | 1737.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1686.00 | 1678.07 | 1703.22 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1832.10 | 1717.88 | 1717.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.70 | 1765.84 | 1747.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1847.10 | 1852.34 | 1807.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-15 10:15:00 | 1865.50 | 1850.20 | 1810.66 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-12-23 09:15:00 | 1797.70 | 1851.57 | 1818.78 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1661.90 | 1793.23 | 1793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 1660.20 | 1790.61 | 1792.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 1739.20 | 1736.04 | 1758.77 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-20 09:15:00 | 1708.10 | 1735.41 | 1757.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1706.10 | 1700.14 | 1730.50 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-04 09:15:00 | 1607.90 | 1700.23 | 1729.51 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1308.00 | 1220.17 | 1317.04 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 13:15:00 | 1319.80 | 1223.84 | 1316.98 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-03-05 13:15:00 | 1547.89 | 2025-03-25 11:15:00 | 1601.60 | EXIT_EMA400 | -53.71 |
| BUY | 2025-07-18 11:15:00 | 1874.50 | 2025-07-24 09:15:00 | 1698.60 | EXIT_EMA400 | -175.90 |
| SELL | 2025-09-05 09:15:00 | 1657.30 | 2025-09-10 09:15:00 | 1754.20 | EXIT_EMA400 | -96.90 |
| BUY | 2025-12-15 10:15:00 | 1865.50 | 2025-12-23 09:15:00 | 1797.70 | EXIT_EMA400 | -67.80 |
| SELL | 2026-01-20 09:15:00 | 1708.10 | 2026-02-06 09:15:00 | 1559.39 | TARGET | 148.71 |
| SELL | 2026-02-04 09:15:00 | 1607.90 | 2026-02-24 09:15:00 | 1243.06 | TARGET | 364.84 |
