# Aavas Financiers Ltd. (AAVAS.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1384.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 9 |
| ENTRY1 | 6 |
| ENTRY2 | 8 |
| EXIT | 6 |

## P&L

- **Trades closed:** 14
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / EMA400 exits:** 3 / 11
- **Total realized P&L (per unit):** 55.90
- **Avg P&L per closed trade:** 3.99

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 1680.95 | 1717.93 | 1718.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 1670.15 | 1717.05 | 1717.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1708.95 | 1706.02 | 1711.27 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 13:15:00 | 1809.80 | 1714.71 | 1714.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-06 09:15:00 | 1869.50 | 1718.20 | 1716.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 11:15:00 | 1807.75 | 1808.97 | 1773.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-01 09:15:00 | 1853.25 | 1809.49 | 1774.85 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 1777.85 | 1811.17 | 1777.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-04 09:15:00 | 1790.80 | 1809.74 | 1777.36 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-04 12:15:00 | 1777.00 | 1809.24 | 1777.59 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 13:15:00 | 1690.00 | 1758.17 | 1758.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 14:15:00 | 1680.10 | 1757.39 | 1758.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 1680.00 | 1679.59 | 1703.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 15:15:00 | 1670.00 | 1679.45 | 1703.35 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1690.95 | 1673.13 | 1693.77 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-13 09:15:00 | 1657.20 | 1673.41 | 1693.20 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1686.35 | 1671.75 | 1690.35 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-19 12:15:00 | 1660.00 | 1672.26 | 1689.71 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 15:15:00 | 1682.00 | 1672.38 | 1689.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-20 09:15:00 | 1677.15 | 1672.42 | 1689.45 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1676.00 | 1670.03 | 1684.41 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 10:15:00 | 1695.00 | 1670.77 | 1684.22 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 1720.55 | 1684.66 | 1684.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 11:15:00 | 1724.80 | 1688.46 | 1686.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 1674.40 | 1693.05 | 1689.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-03-10 09:15:00 | 1823.90 | 1692.54 | 1690.06 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1701.20 | 1697.96 | 1692.92 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-03-11 12:15:00 | 1749.00 | 1698.65 | 1693.34 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1904.05 | 1906.88 | 1823.35 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-04-07 15:15:00 | 2024.95 | 1910.50 | 1827.65 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-02 09:15:00 | 1904.80 | 1998.70 | 1915.75 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1782.50 | 1867.47 | 1867.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 09:15:00 | 1765.10 | 1861.58 | 1864.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 1849.00 | 1843.88 | 1854.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-02 09:15:00 | 1799.70 | 1843.44 | 1854.04 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-06-06 12:15:00 | 1856.30 | 1834.52 | 1847.65 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 09:15:00 | 1967.40 | 1855.13 | 1854.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 10:15:00 | 1976.50 | 1856.34 | 1855.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1901.90 | 1903.71 | 1882.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-08 09:15:00 | 1961.90 | 1903.50 | 1882.98 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1961.90 | 1903.50 | 1882.98 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-08 11:15:00 | 1983.00 | 1904.96 | 1883.92 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-18 11:15:00 | 1891.00 | 1922.91 | 1899.60 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1741.10 | 1884.87 | 1885.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1714.00 | 1877.37 | 1881.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 09:15:00 | 1649.00 | 1648.78 | 1716.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-25 11:15:00 | 1632.20 | 1654.31 | 1702.93 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1672.00 | 1646.47 | 1688.02 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-07 12:15:00 | 1662.50 | 1647.10 | 1687.72 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-23 09:15:00 | 1675.10 | 1632.93 | 1667.93 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-01 09:15:00 | 1853.25 | 2024-10-04 12:15:00 | 1777.00 | EXIT_EMA400 | -76.25 |
| BUY | 2024-10-04 09:15:00 | 1790.80 | 2024-10-04 12:15:00 | 1777.00 | EXIT_EMA400 | -13.80 |
| SELL | 2024-11-29 15:15:00 | 1670.00 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -25.00 |
| SELL | 2024-12-13 09:15:00 | 1657.20 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -37.80 |
| SELL | 2024-12-19 12:15:00 | 1660.00 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -35.00 |
| SELL | 2024-12-20 09:15:00 | 1677.15 | 2025-01-02 10:15:00 | 1695.00 | EXIT_EMA400 | -17.85 |
| BUY | 2025-03-11 12:15:00 | 1749.00 | 2025-03-17 15:15:00 | 1915.98 | TARGET | 166.98 |
| BUY | 2025-03-10 09:15:00 | 1823.90 | 2025-04-22 10:15:00 | 2225.42 | TARGET | 401.52 |
| BUY | 2025-04-07 15:15:00 | 2024.95 | 2025-05-02 09:15:00 | 1904.80 | EXIT_EMA400 | -120.15 |
| SELL | 2025-06-02 09:15:00 | 1799.70 | 2025-06-06 12:15:00 | 1856.30 | EXIT_EMA400 | -56.60 |
| BUY | 2025-07-08 09:15:00 | 1961.90 | 2025-07-18 11:15:00 | 1891.00 | EXIT_EMA400 | -70.90 |
| BUY | 2025-07-08 11:15:00 | 1983.00 | 2025-07-18 11:15:00 | 1891.00 | EXIT_EMA400 | -92.00 |
| SELL | 2025-10-07 12:15:00 | 1662.50 | 2025-10-15 12:15:00 | 1586.84 | TARGET | 75.66 |
| SELL | 2025-09-25 11:15:00 | 1632.20 | 2025-10-23 09:15:00 | 1675.10 | EXIT_EMA400 | -42.90 |
