# Tega Industries Ltd. (TEGA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 1663.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 5 |
| ENTRY2 | 1 |
| EXIT | 5 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 1 / 5
- **Target hits / EMA400 exits:** 1 / 5
- **Total realized P&L (per unit):** -15.62
- **Avg P&L per closed trade:** -2.60

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-14 12:15:00 | 951.65 | 970.59 | 970.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 939.25 | 967.64 | 969.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 14:15:00 | 930.25 | 912.67 | 934.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-10-23 14:15:00 | 883.20 | 915.49 | 933.80 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2023-11-01 11:15:00 | 933.95 | 908.59 | 926.59 | Close above EMA400 |

### Cycle 2 — BUY (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-10 09:15:00 | 1022.20 | 940.56 | 940.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-10 12:15:00 | 1038.35 | 943.30 | 941.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-30 10:15:00 | 989.90 | 991.77 | 972.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-04 09:15:00 | 1015.60 | 991.47 | 973.14 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-05 14:15:00 | 973.20 | 992.96 | 974.98 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 15:15:00 | 1706.65 | 1842.88 | 1843.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 10:15:00 | 1699.65 | 1840.10 | 1841.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 1624.85 | 1622.35 | 1686.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 09:15:00 | 1591.10 | 1628.28 | 1678.15 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-02-05 11:15:00 | 1629.60 | 1570.90 | 1629.53 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 12:15:00 | 1615.20 | 1429.04 | 1428.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1633.40 | 1440.27 | 1433.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 1512.10 | 1519.27 | 1483.96 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-30 09:15:00 | 1556.80 | 1503.92 | 1485.92 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-09-29 11:15:00 | 1882.00 | 1959.25 | 1886.19 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 1880.30 | 1917.17 | 1917.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1863.00 | 1912.67 | 1914.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1837.00 | 1816.32 | 1856.89 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-06 09:15:00 | 1780.00 | 1816.16 | 1855.03 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1832.90 | 1813.43 | 1850.98 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-11 09:15:00 | 1778.80 | 1813.17 | 1849.56 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-25 09:15:00 | 1827.70 | 1749.79 | 1801.59 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2023-10-23 14:15:00 | 883.20 | 2023-11-01 11:15:00 | 933.95 | EXIT_EMA400 | -50.75 |
| BUY | 2023-12-04 09:15:00 | 1015.60 | 2023-12-05 14:15:00 | 973.20 | EXIT_EMA400 | -42.40 |
| SELL | 2025-01-22 09:15:00 | 1591.10 | 2025-02-05 11:15:00 | 1629.60 | EXIT_EMA400 | -38.50 |
| BUY | 2025-06-30 09:15:00 | 1556.80 | 2025-07-07 09:15:00 | 1769.43 | TARGET | 212.63 |
| SELL | 2026-02-06 09:15:00 | 1780.00 | 2026-02-25 09:15:00 | 1827.70 | EXIT_EMA400 | -47.70 |
| SELL | 2026-02-11 09:15:00 | 1778.80 | 2026-02-25 09:15:00 | 1827.70 | EXIT_EMA400 | -48.90 |
