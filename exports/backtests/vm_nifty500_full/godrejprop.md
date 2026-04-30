# Godrej Properties Ltd. (GODREJPROP.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 1835.20
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 8 |
| ENTRY1 | 6 |
| ENTRY2 | 5 |
| EXIT | 6 |

## P&L

- **Trades closed:** 11
- **Trades open at end:** 0
- **Winners / losers:** 3 / 8
- **Target hits / EMA400 exits:** 3 / 8
- **Total realized P&L (per unit):** 340.15
- **Avg P&L per closed trade:** 30.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 2915.80 | 2971.01 | 2971.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 11:15:00 | 2888.90 | 2969.56 | 2970.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 2967.90 | 2928.89 | 2946.96 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 11:15:00 | 2882.50 | 2930.05 | 2946.77 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 2874.15 | 2917.94 | 2938.88 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-09-19 10:15:00 | 2861.15 | 2917.37 | 2938.50 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2913.75 | 2915.25 | 2936.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-09-20 13:15:00 | 2954.25 | 2915.82 | 2936.65 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 3257.60 | 2955.92 | 2955.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 11:15:00 | 3284.50 | 2959.19 | 2957.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 2983.70 | 3037.68 | 3001.51 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-14 13:15:00 | 3087.35 | 3013.86 | 2995.03 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 3039.25 | 3038.07 | 3009.96 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-10-18 10:15:00 | 3088.50 | 3038.57 | 3010.35 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-22 09:15:00 | 3004.55 | 3045.51 | 3015.76 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 2808.70 | 2994.32 | 2994.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 2762.40 | 2980.86 | 2987.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 2830.45 | 2829.47 | 2897.03 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-06 09:15:00 | 2712.25 | 2849.05 | 2873.31 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-23 09:15:00 | 2146.90 | 2056.82 | 2144.11 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 2247.20 | 2158.80 | 2158.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 2265.00 | 2163.83 | 2161.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 10:15:00 | 2331.30 | 2342.50 | 2282.94 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-18 09:15:00 | 2351.40 | 2304.99 | 2281.70 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-25 09:15:00 | 2259.80 | 2315.99 | 2291.63 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 2101.40 | 2270.11 | 2270.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 2083.00 | 2263.40 | 2267.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 2047.00 | 2032.87 | 2098.05 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-09-24 14:15:00 | 2018.60 | 2054.08 | 2095.20 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 2070.90 | 2036.86 | 2074.81 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-08 09:15:00 | 2053.00 | 2037.38 | 2074.70 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2031.50 | 2037.23 | 2073.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-10-09 10:15:00 | 2013.00 | 2036.99 | 2073.04 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 2054.10 | 2036.64 | 2071.79 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-10-10 11:15:00 | 2078.00 | 2037.35 | 2071.80 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 2313.50 | 2096.97 | 2096.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 2329.00 | 2118.87 | 2107.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 2190.50 | 2192.54 | 2152.28 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 2073.00 | 2139.79 | 2139.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 2062.30 | 2139.02 | 2139.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 2063.00 | 2049.20 | 2080.32 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-09 09:15:00 | 2025.30 | 2066.35 | 2084.70 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2025.30 | 2066.35 | 2084.70 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-01-09 11:15:00 | 2000.10 | 2065.10 | 2083.89 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 1872.30 | 1796.10 | 1876.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-18 14:15:00 | 1882.40 | 1798.42 | 1876.36 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-09-16 11:15:00 | 2882.50 | 2024-09-20 13:15:00 | 2954.25 | EXIT_EMA400 | -71.75 |
| SELL | 2024-09-19 10:15:00 | 2861.15 | 2024-09-20 13:15:00 | 2954.25 | EXIT_EMA400 | -93.10 |
| BUY | 2024-10-14 13:15:00 | 3087.35 | 2024-10-22 09:15:00 | 3004.55 | EXIT_EMA400 | -82.80 |
| BUY | 2024-10-18 10:15:00 | 3088.50 | 2024-10-22 09:15:00 | 3004.55 | EXIT_EMA400 | -83.95 |
| SELL | 2025-01-06 09:15:00 | 2712.25 | 2025-01-22 12:15:00 | 2229.07 | TARGET | 483.18 |
| BUY | 2025-07-18 09:15:00 | 2351.40 | 2025-07-25 09:15:00 | 2259.80 | EXIT_EMA400 | -91.60 |
| SELL | 2025-09-24 14:15:00 | 2018.60 | 2025-10-10 11:15:00 | 2078.00 | EXIT_EMA400 | -59.40 |
| SELL | 2025-10-08 09:15:00 | 2053.00 | 2025-10-10 11:15:00 | 2078.00 | EXIT_EMA400 | -25.00 |
| SELL | 2025-10-09 10:15:00 | 2013.00 | 2025-10-10 11:15:00 | 2078.00 | EXIT_EMA400 | -65.00 |
| SELL | 2026-01-09 09:15:00 | 2025.30 | 2026-01-19 09:15:00 | 1847.10 | TARGET | 178.20 |
| SELL | 2026-01-09 11:15:00 | 2000.10 | 2026-01-20 11:15:00 | 1748.73 | TARGET | 251.37 |
