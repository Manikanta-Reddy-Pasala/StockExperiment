# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1549.00
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -152.85
- **Avg P&L per closed trade:** -25.48

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1540.10 | 1761.54 | 1762.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 1511.00 | 1759.05 | 1760.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 1541.05 | 1501.94 | 1576.47 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-13 09:15:00 | 1477.25 | 1515.00 | 1569.67 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 1500.00 | 1470.55 | 1523.18 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-30 09:15:00 | 1526.75 | 1471.11 | 1523.20 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 1600.20 | 1526.49 | 1526.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 10:15:00 | 1636.65 | 1536.32 | 1531.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1553.55 | 1651.12 | 1604.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-07 12:15:00 | 1581.40 | 1648.78 | 1604.29 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 12:15:00 | 1581.40 | 1648.78 | 1604.29 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-04-07 13:15:00 | 1599.85 | 1648.29 | 1604.27 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1847.90 | 1943.69 | 1943.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1826.20 | 1942.52 | 1943.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.93 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-19 12:15:00 | 1872.10 | 1928.28 | 1935.37 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-09-11 11:15:00 | 1883.60 | 1833.97 | 1873.93 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1942.00 | 1885.99 | 1885.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1962.30 | 1891.80 | 1888.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-11-11 09:15:00 | 1928.50 | 1912.51 | 1900.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-11-19 09:15:00 | 1884.90 | 1924.51 | 1909.82 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 1803.10 | 1898.20 | 1898.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1793.00 | 1896.27 | 1897.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-02 09:15:00 | 1861.60 | 1889.66 | 1893.76 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-03 12:15:00 | 1897.30 | 1887.57 | 1892.49 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1932.00 | 1896.29 | 1896.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1946.80 | 1896.80 | 1896.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1830.10 | 1895.88 | 1896.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 1820.50 | 1895.13 | 1895.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-23 14:15:00 | 1867.90 | 1876.79 | 1885.55 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1884.00 | 1876.73 | 1885.35 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-24 13:15:00 | 1898.90 | 1877.00 | 1885.40 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-01-13 09:15:00 | 1477.25 | 2025-01-30 09:15:00 | 1526.75 | EXIT_EMA400 | -49.50 |
| BUY | 2025-04-07 12:15:00 | 1581.40 | 2025-04-07 13:15:00 | 1599.85 | EXIT_EMA400 | 18.45 |
| SELL | 2025-08-19 12:15:00 | 1872.10 | 2025-09-11 11:15:00 | 1883.60 | EXIT_EMA400 | -11.50 |
| BUY | 2025-11-11 09:15:00 | 1928.50 | 2025-11-19 09:15:00 | 1884.90 | EXIT_EMA400 | -43.60 |
| SELL | 2025-12-02 09:15:00 | 1861.60 | 2025-12-03 12:15:00 | 1897.30 | EXIT_EMA400 | -35.70 |
| SELL | 2025-12-23 14:15:00 | 1867.90 | 2025-12-24 13:15:00 | 1898.90 | EXIT_EMA400 | -31.00 |
