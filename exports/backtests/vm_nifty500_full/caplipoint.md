# Caplin Point Laboratories Ltd. (CAPLIPOINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 1709.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 5 |
| ENTRY1 | 6 |
| ENTRY2 | 4 |
| EXIT | 6 |

## P&L

- **Trades closed:** 10
- **Trades open at end:** 0
- **Winners / losers:** 4 / 6
- **Target hits / EMA400 exits:** 4 / 6
- **Total realized P&L (per unit):** 349.55
- **Avg P&L per closed trade:** 34.96

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 13:15:00 | 1265.00 | 1381.26 | 1381.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-26 11:15:00 | 1252.55 | 1368.53 | 1375.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 1371.10 | 1353.02 | 1366.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-02 09:15:00 | 1317.95 | 1354.16 | 1366.38 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1317.95 | 1354.16 | 1366.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-04 11:15:00 | 1306.40 | 1348.25 | 1362.37 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1339.00 | 1328.05 | 1345.60 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-04-24 13:15:00 | 1327.10 | 1328.81 | 1345.05 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-04-25 10:15:00 | 1356.95 | 1329.32 | 1344.98 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 10:15:00 | 1456.00 | 1333.29 | 1333.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 11:15:00 | 1469.90 | 1334.65 | 1333.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 1507.80 | 1514.68 | 1464.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-12 09:15:00 | 1547.35 | 1513.09 | 1466.08 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-23 14:15:00 | 1824.95 | 1902.99 | 1835.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-02-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 15:15:00 | 2031.80 | 2198.24 | 2198.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 2017.65 | 2162.82 | 2179.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 13:15:00 | 1959.85 | 1957.52 | 2031.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-04-02 09:15:00 | 1946.30 | 1977.02 | 2025.45 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-03 09:15:00 | 2036.90 | 1977.26 | 2023.90 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2271.90 | 1979.18 | 1978.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 12:15:00 | 2286.00 | 1988.06 | 1982.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 2055.70 | 2081.86 | 2041.83 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-11 14:15:00 | 2114.20 | 2077.82 | 2043.30 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2099.70 | 2085.92 | 2052.57 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-19 10:15:00 | 2051.60 | 2085.66 | 2053.74 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 1930.00 | 2052.88 | 2053.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1900.90 | 2047.80 | 2050.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2069.00 | 2043.86 | 2048.55 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 2139.50 | 2053.10 | 2052.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 2143.10 | 2058.04 | 2055.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 2086.70 | 2098.42 | 2079.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-28 09:15:00 | 2119.80 | 2098.50 | 2079.32 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 2119.80 | 2098.50 | 2079.32 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-08-28 10:15:00 | 2143.40 | 2098.94 | 2079.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2115.50 | 2101.91 | 2082.40 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-09-02 15:15:00 | 2125.80 | 2103.73 | 2084.56 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-09-26 09:15:00 | 2080.90 | 2206.41 | 2157.04 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 2038.50 | 2120.37 | 2120.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 2009.50 | 2111.62 | 2116.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1952.60 | 1949.94 | 1996.18 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 12:15:00 | 1919.80 | 1948.94 | 1992.77 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-23 09:15:00 | 1977.90 | 1941.86 | 1975.06 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-02 09:15:00 | 1317.95 | 2024-04-25 10:15:00 | 1356.95 | EXIT_EMA400 | -39.00 |
| SELL | 2024-04-04 11:15:00 | 1306.40 | 2024-04-25 10:15:00 | 1356.95 | EXIT_EMA400 | -50.55 |
| SELL | 2024-04-24 13:15:00 | 1327.10 | 2024-04-25 10:15:00 | 1356.95 | EXIT_EMA400 | -29.85 |
| BUY | 2024-08-12 09:15:00 | 1547.35 | 2024-08-19 11:15:00 | 1791.15 | TARGET | 243.80 |
| SELL | 2025-04-02 09:15:00 | 1946.30 | 2025-04-03 09:15:00 | 2036.90 | EXIT_EMA400 | -90.60 |
| BUY | 2025-06-11 14:15:00 | 2114.20 | 2025-06-19 10:15:00 | 2051.60 | EXIT_EMA400 | -62.60 |
| BUY | 2025-08-28 09:15:00 | 2119.80 | 2025-09-11 09:15:00 | 2241.25 | TARGET | 121.45 |
| BUY | 2025-09-02 15:15:00 | 2125.80 | 2025-09-11 09:15:00 | 2249.51 | TARGET | 123.71 |
| BUY | 2025-08-28 10:15:00 | 2143.40 | 2025-09-17 10:15:00 | 2334.69 | TARGET | 191.29 |
| SELL | 2025-12-08 12:15:00 | 1919.80 | 2025-12-23 09:15:00 | 1977.90 | EXIT_EMA400 | -58.10 |
