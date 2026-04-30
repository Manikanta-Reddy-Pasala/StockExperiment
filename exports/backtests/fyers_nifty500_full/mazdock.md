# Mazagoan Dock Shipbuilders Ltd. (MAZDOCK.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2738.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 1 |
| EXIT | 4 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 0 / 5
- **Target hits / EMA400 exits:** 0 / 5
- **Total realized P&L (per unit):** -608.87
- **Avg P&L per closed trade:** -121.77

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 2099.00 | 2176.09 | 2176.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 2065.15 | 2174.21 | 2175.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 10:15:00 | 2192.57 | 2131.98 | 2151.92 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 14:15:00 | 2337.53 | 2166.65 | 2166.30 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 2103.00 | 2165.49 | 2165.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 2036.45 | 2157.49 | 2161.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 2130.45 | 2108.37 | 2133.19 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-13 09:15:00 | 1974.43 | 2102.26 | 2125.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-26 09:15:00 | 2198.35 | 2068.92 | 2101.54 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 13:15:00 | 2326.60 | 2128.60 | 2128.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 2394.48 | 2135.61 | 2131.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-23 09:15:00 | 2320.00 | 2342.63 | 2258.62 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-20 11:15:00 | 2475.80 | 2266.68 | 2250.93 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 2274.85 | 2279.69 | 2258.58 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-22 10:15:00 | 2248.45 | 2279.37 | 2258.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 2088.00 | 2258.81 | 2259.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 2019.25 | 2246.78 | 2253.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 09:15:00 | 2227.30 | 2189.52 | 2218.11 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 14:15:00 | 2376.05 | 2237.26 | 2236.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 2438.85 | 2240.64 | 2238.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2282.00 | 2457.72 | 2370.67 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-04-08 09:15:00 | 2389.20 | 2448.55 | 2368.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-04-08 10:15:00 | 2362.00 | 2447.69 | 2368.95 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 2662.90 | 3092.41 | 3094.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2639.00 | 2942.30 | 3007.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 14:15:00 | 2758.30 | 2757.26 | 2850.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 14:15:00 | 2730.60 | 2822.12 | 2846.05 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 2808.50 | 2775.34 | 2811.45 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-11-19 09:15:00 | 2775.00 | 2778.47 | 2810.64 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-11-20 09:15:00 | 2818.00 | 2778.83 | 2809.71 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 2670.70 | 2408.86 | 2407.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 2688.00 | 2426.39 | 2416.55 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-13 09:15:00 | 1974.43 | 2024-11-26 09:15:00 | 2198.35 | EXIT_EMA400 | -223.92 |
| BUY | 2025-01-20 11:15:00 | 2475.80 | 2025-01-22 10:15:00 | 2248.45 | EXIT_EMA400 | -227.35 |
| BUY | 2025-04-08 09:15:00 | 2389.20 | 2025-04-08 10:15:00 | 2362.00 | EXIT_EMA400 | -27.20 |
| SELL | 2025-10-31 14:15:00 | 2730.60 | 2025-11-20 09:15:00 | 2818.00 | EXIT_EMA400 | -87.40 |
| SELL | 2025-11-19 09:15:00 | 2775.00 | 2025-11-20 09:15:00 | 2818.00 | EXIT_EMA400 | -43.00 |
