# Aavas Financiers Ltd. (AAVAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1380.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT3 | 12 |
| ENTRY1 | 8 |
| ENTRY2 | 9 |
| EXIT | 8 |

## P&L

- **Trades closed:** 17
- **Trades open at end:** 0
- **Winners / losers:** 3 / 14
- **Total realized P&L (per unit):** -390.15
- **Avg P&L per closed trade:** -22.95

## Last 30 Signals

| Time | Type | Trend | Price | EMA200 | EMA400 | Note |
|------|------|-------|-------|--------|--------|------|
| 2025-01-02 09:15:00 | ALERT3 | SELL | 1677.15 | 1670.55 | 1684.03 | EMA400 retest candle locked |
| 2025-01-02 10:15:00 | EXIT | SELL | 1695.00 | 1670.80 | 1684.08 | Close above EMA400 |
| 2025-02-04 13:15:00 | CROSSOVER | BUY | 1710.30 | 1684.48 | 1684.41 | EMA200 above EMA400 |
| 2025-02-06 10:15:00 | ALERT1 | BUY | 1722.35 | 1687.07 | 1685.74 | Break + close above crossover candle high |
| 2025-02-11 09:15:00 | ALERT2 | BUY | 1673.35 | 1692.19 | 1688.57 | EMA200 retest candle locked |
| 2025-03-10 09:15:00 | ENTRY1 | BUY | 1823.90 | 1692.22 | 1689.74 | Buy entry 1 (retest1 break) |
| 2025-03-11 09:15:00 | ALERT3 | BUY | 1701.20 | 1697.66 | 1692.61 | EMA400 retest candle locked |
| 2025-03-11 12:15:00 | ENTRY2 | BUY | 1749.00 | 1698.36 | 1693.04 | Buy entry 2 (retest2 break) |
| 2025-04-07 09:15:00 | ALERT3 | BUY | 1904.05 | 1906.63 | 1823.13 | EMA400 retest candle locked |
| 2025-04-07 15:15:00 | ENTRY2 | BUY | 2024.95 | 1910.28 | 1827.44 | Buy entry 2 (retest2 break) |
| 2025-05-02 09:15:00 | EXIT | BUY | 1904.80 | 1998.55 | 1915.59 | Close below EMA400 |
| 2025-05-23 09:15:00 | CROSSOVER | SELL | 1782.50 | 1867.37 | 1867.43 | EMA200 below EMA400 |
| 2025-05-26 09:15:00 | ALERT1 | SELL | 1765.10 | 1861.49 | 1864.44 | Break + close below crossover candle low |
| 2025-05-30 14:15:00 | ALERT2 | SELL | 1849.00 | 1843.83 | 1854.28 | EMA200 retest candle locked |
| 2025-06-02 09:15:00 | ENTRY1 | SELL | 1799.70 | 1843.39 | 1853.95 | Sell entry 1 (retest1 break) |
| 2025-06-06 12:15:00 | EXIT | SELL | 1856.30 | 1834.62 | 1847.65 | Close above EMA400 |
| 2025-06-27 09:15:00 | CROSSOVER | BUY | 1967.40 | 1855.25 | 1854.98 | EMA200 above EMA400 |
| 2025-06-27 10:15:00 | ALERT1 | BUY | 1976.50 | 1856.46 | 1855.58 | Break + close above crossover candle high |
| 2025-07-07 10:15:00 | ALERT2 | BUY | 1901.90 | 1903.77 | 1882.48 | EMA200 retest candle locked |
| 2025-07-08 09:15:00 | ENTRY1 | BUY | 1961.90 | 1903.56 | 1883.00 | Buy entry 1 (retest1 break) |
| 2025-07-08 09:15:00 | ALERT3 | BUY | 1961.90 | 1903.56 | 1883.00 | EMA400 retest candle locked |
| 2025-07-08 11:15:00 | ENTRY2 | BUY | 1982.30 | 1905.02 | 1883.93 | Buy entry 2 (retest2 break) |
| 2025-07-18 11:15:00 | EXIT | BUY | 1891.00 | 1922.97 | 1899.63 | Close below EMA400 |
| 2025-07-31 09:15:00 | CROSSOVER | SELL | 1741.30 | 1884.94 | 1885.23 | EMA200 below EMA400 |
| 2025-07-31 14:15:00 | ALERT1 | SELL | 1715.20 | 1877.44 | 1881.42 | Break + close below crossover candle low |
| 2025-09-15 09:15:00 | ALERT2 | SELL | 1650.00 | 1648.79 | 1716.89 | EMA200 retest candle locked |
| 2025-09-25 11:15:00 | ENTRY1 | SELL | 1632.20 | 1654.30 | 1702.92 | Sell entry 1 (retest1 break) |
| 2025-10-07 09:15:00 | ALERT3 | SELL | 1672.00 | 1646.36 | 1687.95 | EMA400 retest candle locked |
| 2025-10-07 12:15:00 | ENTRY2 | SELL | 1662.50 | 1646.99 | 1687.65 | Sell entry 2 (retest2 break) |
| 2025-10-23 09:15:00 | EXIT | SELL | 1676.00 | 1632.82 | 1667.85 | Close above EMA400 |

## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | P&L |
|-------|-----------|-------|-----------|------|-----|
| SELL | 2024-02-29 14:15:00 | 1454.25 | 2024-04-04 09:15:00 | 1472.20 | -17.95 |
| SELL | 2024-04-02 09:15:00 | 1404.00 | 2024-04-04 09:15:00 | 1472.20 | -68.20 |
| BUY | 2024-05-31 14:15:00 | 1598.95 | 2024-06-04 10:15:00 | 1520.20 | -78.75 |
| BUY | 2024-09-03 09:15:00 | 1723.05 | 2024-10-04 14:15:00 | 1770.00 | 46.95 |
| BUY | 2024-10-04 09:15:00 | 1790.80 | 2024-10-04 14:15:00 | 1770.00 | -20.80 |
| SELL | 2024-11-29 14:15:00 | 1674.90 | 2025-01-02 10:15:00 | 1695.00 | -20.10 |
| SELL | 2024-12-12 11:15:00 | 1676.10 | 2025-01-02 10:15:00 | 1695.00 | -18.90 |
| SELL | 2024-12-19 12:15:00 | 1660.00 | 2025-01-02 10:15:00 | 1695.00 | -35.00 |
| SELL | 2024-12-20 09:15:00 | 1677.15 | 2025-01-02 10:15:00 | 1695.00 | -17.85 |
| BUY | 2025-03-10 09:15:00 | 1823.90 | 2025-05-02 09:15:00 | 1904.80 | 80.90 |
| BUY | 2025-03-11 12:15:00 | 1749.00 | 2025-05-02 09:15:00 | 1904.80 | 155.80 |
| BUY | 2025-04-07 15:15:00 | 2024.95 | 2025-05-02 09:15:00 | 1904.80 | -120.15 |
| SELL | 2025-06-02 09:15:00 | 1799.70 | 2025-06-06 12:15:00 | 1856.30 | -56.60 |
| BUY | 2025-07-08 09:15:00 | 1961.90 | 2025-07-18 11:15:00 | 1891.00 | -70.90 |
| BUY | 2025-07-08 11:15:00 | 1982.30 | 2025-07-18 11:15:00 | 1891.00 | -91.30 |
| SELL | 2025-09-25 11:15:00 | 1632.20 | 2025-10-23 09:15:00 | 1676.00 | -43.80 |
| SELL | 2025-10-07 12:15:00 | 1662.50 | 2025-10-23 09:15:00 | 1676.00 | -13.50 |
