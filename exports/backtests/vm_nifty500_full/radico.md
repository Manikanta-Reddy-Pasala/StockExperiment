# Radico Khaitan Ltd (RADICO.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 3423.10
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 7 |
| ENTRY1 | 8 |
| ENTRY2 | 1 |
| EXIT | 8 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 3 / 6
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 81.65
- **Avg P&L per closed trade:** 9.07

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-08 13:15:00 | 1254.00 | 1288.99 | 1289.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-11 09:15:00 | 1249.00 | 1287.93 | 1288.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 09:15:00 | 1239.25 | 1226.04 | 1248.31 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-09 15:15:00 | 1212.80 | 1226.02 | 1247.64 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-10-11 09:15:00 | 1250.65 | 1225.78 | 1246.67 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 09:15:00 | 1391.15 | 1251.86 | 1251.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 10:15:00 | 1401.35 | 1253.35 | 1252.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 1614.10 | 1620.90 | 1543.13 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-19 09:15:00 | 1627.60 | 1619.55 | 1546.59 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1632.60 | 1681.94 | 1626.26 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-02-22 10:15:00 | 1624.45 | 1681.37 | 1626.25 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 11:15:00 | 1552.85 | 1599.47 | 1599.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 09:15:00 | 1642.75 | 1600.03 | 1599.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-20 10:15:00 | 1651.45 | 1600.55 | 1600.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 13:15:00 | 1603.85 | 1604.89 | 1602.43 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-22 14:15:00 | 1650.00 | 1605.34 | 1602.67 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1680.75 | 1716.21 | 1680.43 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-05-07 14:15:00 | 1705.30 | 1715.87 | 1680.61 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1683.10 | 1714.93 | 1680.84 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-05-08 12:15:00 | 1680.30 | 1714.59 | 1680.84 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 1578.35 | 1664.73 | 1664.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1537.45 | 1658.53 | 1661.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1705.60 | 1653.31 | 1658.88 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 11:15:00 | 1696.00 | 1664.18 | 1664.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 14:15:00 | 1713.75 | 1665.43 | 1664.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 1726.35 | 1738.07 | 1709.69 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-26 09:15:00 | 1754.00 | 1709.35 | 1702.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1710.30 | 1715.69 | 1707.24 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-02 09:15:00 | 1682.05 | 1715.36 | 1707.11 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 1678.45 | 1700.69 | 1700.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1666.00 | 1700.12 | 1700.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 12:15:00 | 1695.20 | 1692.92 | 1696.61 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 1765.00 | 1700.32 | 1700.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 1781.15 | 1701.12 | 1700.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 2017.30 | 2020.88 | 1924.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-09 14:15:00 | 2112.10 | 2017.77 | 1934.39 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2391.55 | 2469.17 | 2373.56 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-10 09:15:00 | 2328.80 | 2463.24 | 2373.83 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 2075.75 | 2317.40 | 2317.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 2047.00 | 2251.32 | 2278.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 2203.20 | 2186.68 | 2239.57 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-20 15:15:00 | 2139.00 | 2186.21 | 2239.07 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-06 14:15:00 | 2210.65 | 2133.48 | 2194.18 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 2358.00 | 2222.15 | 2221.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 2426.00 | 2227.69 | 2224.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2233.45 | 2260.44 | 2242.87 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2288.00 | 2259.60 | 2243.05 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 2538.40 | 2600.18 | 2534.03 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-02 12:15:00 | 2527.00 | 2598.81 | 2534.00 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 2932.30 | 3139.53 | 3140.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2903.30 | 3133.03 | 3136.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 2772.70 | 2750.79 | 2858.74 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-09 09:15:00 | 2677.90 | 2750.41 | 2857.48 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 2844.40 | 2753.88 | 2850.00 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-03-11 14:15:00 | 2880.20 | 2755.14 | 2850.15 | Close above EMA400 |

### Cycle 12 — BUY (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 09:15:00 | 3258.10 | 2824.52 | 2824.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 11:15:00 | 3281.70 | 2833.37 | 2828.91 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-09 15:15:00 | 1212.80 | 2023-10-11 09:15:00 | 1250.65 | EXIT_EMA400 | -37.85 |
| BUY | 2024-01-19 09:15:00 | 1627.60 | 2024-02-22 10:15:00 | 1624.45 | EXIT_EMA400 | -3.15 |
| BUY | 2024-03-22 14:15:00 | 1650.00 | 2024-04-12 13:15:00 | 1792.00 | TARGET | 142.00 |
| BUY | 2024-05-07 14:15:00 | 1705.30 | 2024-05-08 12:15:00 | 1680.30 | EXIT_EMA400 | -25.00 |
| BUY | 2024-07-26 09:15:00 | 1754.00 | 2024-08-02 09:15:00 | 1682.05 | EXIT_EMA400 | -71.95 |
| BUY | 2024-10-09 14:15:00 | 2112.10 | 2025-01-10 09:15:00 | 2328.80 | EXIT_EMA400 | 216.70 |
| SELL | 2025-02-20 15:15:00 | 2139.00 | 2025-03-06 14:15:00 | 2210.65 | EXIT_EMA400 | -71.65 |
| BUY | 2025-04-08 09:15:00 | 2288.00 | 2025-04-15 09:15:00 | 2422.85 | TARGET | 134.85 |
| SELL | 2026-03-09 09:15:00 | 2677.90 | 2026-03-11 14:15:00 | 2880.20 | EXIT_EMA400 | -202.30 |
