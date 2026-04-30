# Muthoot Finance Ltd. (MUTHOOTFIN.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 3424.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT3 | 6 |
| ENTRY1 | 6 |
| ENTRY2 | 3 |
| EXIT | 6 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 6 / 3
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** 33.65
- **Avg P&L per closed trade:** 3.74

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 10:15:00 | 1195.60 | 1263.43 | 1263.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 11:15:00 | 1188.60 | 1258.22 | 1260.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-17 14:15:00 | 1250.45 | 1248.30 | 1254.61 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 1314.95 | 1259.52 | 1259.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 1316.65 | 1260.09 | 1259.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-10 09:15:00 | 1242.55 | 1285.64 | 1274.26 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-11-10 10:15:00 | 1260.10 | 1285.38 | 1274.19 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 1260.10 | 1285.38 | 1274.19 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-11-10 11:15:00 | 1260.60 | 1285.14 | 1274.12 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 13:15:00 | 1341.80 | 1398.11 | 1398.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 1329.10 | 1386.69 | 1392.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 09:15:00 | 1388.85 | 1353.55 | 1371.65 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-03-05 14:15:00 | 1357.00 | 1355.10 | 1371.99 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 14:15:00 | 1357.00 | 1355.10 | 1371.99 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-03-06 09:15:00 | 1377.95 | 1355.40 | 1371.97 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 09:15:00 | 1478.90 | 1378.68 | 1378.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 1491.95 | 1380.83 | 1379.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-09 09:15:00 | 1582.95 | 1617.57 | 1544.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-09 11:15:00 | 1632.20 | 1617.38 | 1545.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 1613.10 | 1672.99 | 1612.66 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 09:15:00 | 1687.65 | 1672.31 | 1613.80 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-07-23 14:15:00 | 1735.00 | 1792.23 | 1739.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 1819.80 | 1915.75 | 1916.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 1805.35 | 1913.64 | 1915.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 1894.95 | 1891.16 | 1903.04 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 15:15:00 | 1947.80 | 1910.30 | 1910.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 1958.55 | 1912.66 | 1911.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 14:15:00 | 2096.60 | 2097.92 | 2037.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-14 10:15:00 | 2135.40 | 2098.36 | 2038.78 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 2161.65 | 2204.88 | 2150.71 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-02-24 12:15:00 | 2170.75 | 2204.54 | 2150.81 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-02-28 09:15:00 | 2144.95 | 2203.28 | 2154.78 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 2136.10 | 2199.06 | 2199.28 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 13:15:00 | 2245.50 | 2199.45 | 2199.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 09:15:00 | 2297.10 | 2201.37 | 2200.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 2205.30 | 2212.38 | 2206.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2245.30 | 2213.28 | 2206.86 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 2209.40 | 2216.61 | 2208.94 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-05-13 14:15:00 | 2214.80 | 2216.60 | 2208.97 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-05-15 09:15:00 | 2148.00 | 2218.51 | 2210.29 | Close below EMA400 |

### Cycle 9 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 2096.60 | 2201.85 | 2202.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 12:15:00 | 2088.10 | 2199.62 | 2201.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 2151.00 | 2145.94 | 2169.79 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 2446.20 | 2189.55 | 2188.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 2508.50 | 2192.72 | 2190.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 10:15:00 | 2597.40 | 2606.99 | 2508.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-29 12:15:00 | 2625.10 | 2607.20 | 2509.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 2545.10 | 2611.82 | 2540.72 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-08-12 11:15:00 | 2539.50 | 2611.10 | 2540.71 | Close below EMA400 |

### Cycle 11 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 3489.50 | 3671.97 | 3672.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 11:15:00 | 3444.40 | 3652.96 | 3662.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3447.20 | 3330.08 | 3434.44 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-11-10 10:15:00 | 1260.10 | 2023-11-10 11:15:00 | 1260.60 | EXIT_EMA400 | 0.50 |
| SELL | 2024-03-05 14:15:00 | 1357.00 | 2024-03-06 09:15:00 | 1377.95 | EXIT_EMA400 | -20.95 |
| BUY | 2024-05-09 11:15:00 | 1632.20 | 2024-07-23 14:15:00 | 1735.00 | EXIT_EMA400 | 102.80 |
| BUY | 2024-06-05 09:15:00 | 1687.65 | 2024-07-23 14:15:00 | 1735.00 | EXIT_EMA400 | 47.35 |
| BUY | 2025-02-24 12:15:00 | 2170.75 | 2025-02-27 09:15:00 | 2230.57 | TARGET | 59.82 |
| BUY | 2025-01-14 10:15:00 | 2135.40 | 2025-02-28 09:15:00 | 2144.95 | EXIT_EMA400 | 9.55 |
| BUY | 2025-05-13 14:15:00 | 2214.80 | 2025-05-14 09:15:00 | 2232.28 | TARGET | 17.48 |
| BUY | 2025-05-12 09:15:00 | 2245.30 | 2025-05-15 09:15:00 | 2148.00 | EXIT_EMA400 | -97.30 |
| BUY | 2025-07-29 12:15:00 | 2625.10 | 2025-08-12 11:15:00 | 2539.50 | EXIT_EMA400 | -85.60 |
