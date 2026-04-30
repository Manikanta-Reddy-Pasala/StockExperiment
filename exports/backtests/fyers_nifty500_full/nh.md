# Narayana Hrudayalaya Ltd. (NH.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 1773.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 50.97
- **Avg P&L per closed trade:** 5.66

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 15:15:00 | 1232.65 | 1239.35 | 1239.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 1222.65 | 1239.19 | 1239.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 10:15:00 | 1232.90 | 1222.40 | 1230.04 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 1286.50 | 1236.76 | 1236.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 1317.60 | 1250.32 | 1244.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 1284.55 | 1284.92 | 1265.43 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 1230.00 | 1255.27 | 1255.29 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1284.80 | 1254.36 | 1254.28 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 1222.20 | 1254.54 | 1254.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1216.50 | 1253.25 | 1253.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 1251.45 | 1250.06 | 1252.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-01 18:15:00 | 1235.00 | 1250.82 | 1252.51 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 1235.00 | 1250.82 | 1252.51 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 09:15:00 | 1209.70 | 1250.41 | 1252.29 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 1245.65 | 1244.11 | 1248.80 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-11-06 15:15:00 | 1249.75 | 1244.16 | 1248.80 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 1295.35 | 1253.09 | 1252.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 11:15:00 | 1318.10 | 1259.99 | 1257.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 11:15:00 | 1273.00 | 1276.37 | 1266.72 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-12-12 10:15:00 | 1297.80 | 1276.85 | 1267.24 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 1285.65 | 1281.74 | 1271.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-19 10:15:00 | 1288.80 | 1281.81 | 1271.49 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 1279.50 | 1282.70 | 1272.60 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-12-23 10:15:00 | 1307.95 | 1282.95 | 1272.78 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1293.50 | 1288.23 | 1276.88 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-12-30 14:15:00 | 1274.65 | 1288.25 | 1277.18 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 13:15:00 | 1817.50 | 1869.56 | 1869.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1803.60 | 1865.56 | 1867.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 13:15:00 | 1836.00 | 1835.15 | 1850.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-05 09:15:00 | 1815.40 | 1834.91 | 1849.86 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1806.70 | 1807.64 | 1830.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-09-18 09:15:00 | 1794.50 | 1807.46 | 1829.78 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-10-06 09:15:00 | 1804.40 | 1774.90 | 1803.39 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 1953.50 | 1796.49 | 1796.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 1996.70 | 1801.44 | 1798.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1893.80 | 1893.99 | 1858.03 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-08 14:15:00 | 1923.40 | 1894.45 | 1858.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1860.90 | 1893.85 | 1860.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-12-10 13:15:00 | 1857.20 | 1893.48 | 1860.39 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1723.50 | 1866.96 | 1867.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1713.60 | 1863.97 | 1865.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 10:15:00 | 1804.70 | 1803.23 | 1829.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-02 12:15:00 | 1783.60 | 1825.96 | 1833.29 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1725.40 | 1701.09 | 1746.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 1749.90 | 1704.48 | 1745.63 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-01 18:15:00 | 1235.00 | 2024-11-06 15:15:00 | 1249.75 | EXIT_EMA400 | -14.75 |
| SELL | 2024-11-04 09:15:00 | 1209.70 | 2024-11-06 15:15:00 | 1249.75 | EXIT_EMA400 | -40.05 |
| BUY | 2024-12-12 10:15:00 | 1297.80 | 2024-12-30 14:15:00 | 1274.65 | EXIT_EMA400 | -23.15 |
| BUY | 2024-12-19 10:15:00 | 1288.80 | 2024-12-30 14:15:00 | 1274.65 | EXIT_EMA400 | -14.15 |
| BUY | 2024-12-23 10:15:00 | 1307.95 | 2024-12-30 14:15:00 | 1274.65 | EXIT_EMA400 | -33.30 |
| SELL | 2025-09-05 09:15:00 | 1815.40 | 2025-09-26 09:15:00 | 1712.01 | TARGET | 103.39 |
| SELL | 2025-09-18 09:15:00 | 1794.50 | 2025-10-06 09:15:00 | 1804.40 | EXIT_EMA400 | -9.90 |
| BUY | 2025-12-08 14:15:00 | 1923.40 | 2025-12-10 13:15:00 | 1857.20 | EXIT_EMA400 | -66.20 |
| SELL | 2026-03-02 12:15:00 | 1783.60 | 2026-03-16 10:15:00 | 1634.52 | TARGET | 149.08 |
