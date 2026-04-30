# Adani Enterprises Ltd. (ADANIENT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 2408.40
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 16 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT3 | 2 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** 179.41
- **Avg P&L per closed trade:** 29.90

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 2433.05 | 2481.57 | 2481.63 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 11:15:00 | 2519.45 | 2481.84 | 2481.72 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 2455.95 | 2481.61 | 2481.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 2454.55 | 2481.34 | 2481.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 09:15:00 | 2373.20 | 2268.03 | 2335.39 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 2891.00 | 2383.34 | 2383.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 13:15:00 | 2924.20 | 2546.51 | 2472.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 14:15:00 | 2895.00 | 2908.16 | 2769.55 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-01-29 09:15:00 | 3046.70 | 2905.26 | 2778.69 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 3033.35 | 3197.75 | 3076.29 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 12:15:00 | 2835.00 | 3088.49 | 3089.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 2816.20 | 3057.67 | 3073.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-14 12:15:00 | 3027.80 | 3017.76 | 3050.69 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-16 13:15:00 | 2981.00 | 3020.87 | 3049.95 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-17 09:15:00 | 3058.10 | 3021.66 | 3049.91 | Close above EMA400 |

### Cycle 6 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 3385.60 | 3074.56 | 3073.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 14:15:00 | 3410.90 | 3136.60 | 3107.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3121.05 | 3176.00 | 3128.95 | EMA200 retest candle locked |

### Cycle 7 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 3064.50 | 3145.00 | 3145.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 3037.35 | 3143.93 | 3144.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 10:15:00 | 3105.00 | 3099.62 | 3119.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-07-29 12:15:00 | 3089.90 | 3099.53 | 3119.59 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-07-30 11:15:00 | 3119.60 | 3099.04 | 3118.74 | Close above EMA400 |

### Cycle 8 — BUY (started 2024-10-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 13:15:00 | 3128.80 | 3077.40 | 3077.34 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-10-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 10:15:00 | 3006.60 | 3077.15 | 3077.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 2998.90 | 3075.78 | 3076.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-30 11:15:00 | 2980.00 | 2975.07 | 3020.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-30 15:15:00 | 2959.00 | 2974.78 | 3019.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-06 11:15:00 | 3022.65 | 2964.08 | 3008.34 | Close above EMA400 |

### Cycle 10 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 2451.50 | 2342.97 | 2342.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2478.70 | 2347.46 | 2345.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 15:15:00 | 2349.00 | 2353.29 | 2348.48 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 2255.00 | 2344.08 | 2344.24 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 2387.20 | 2344.51 | 2344.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 10:15:00 | 2416.70 | 2345.23 | 2344.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 2454.80 | 2463.35 | 2420.16 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-04 10:15:00 | 2478.00 | 2463.50 | 2420.45 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 2459.00 | 2501.24 | 2456.09 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 13:15:00 | 2452.50 | 2500.76 | 2456.07 | Close below EMA400 |

### Cycle 13 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 2265.20 | 2507.17 | 2507.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 2240.50 | 2504.52 | 2506.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 2381.20 | 2343.88 | 2393.71 | EMA200 retest candle locked |

### Cycle 14 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 2616.50 | 2422.51 | 2422.42 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 2456.80 | 2469.85 | 2469.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 15:15:00 | 2446.10 | 2468.80 | 2469.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 13:15:00 | 2279.60 | 2277.23 | 2331.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-01-06 10:15:00 | 2265.00 | 2277.60 | 2328.52 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2182.00 | 2128.62 | 2214.71 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 2215.90 | 2130.28 | 2214.69 | Close above EMA400 |

### Cycle 16 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2101.36 | 2100.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2394.20 | 2106.45 | 2103.03 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-01-29 09:15:00 | 3046.70 | 2024-03-13 09:15:00 | 3033.35 | EXIT_EMA400 | -13.35 |
| SELL | 2024-05-16 13:15:00 | 2981.00 | 2024-05-17 09:15:00 | 3058.10 | EXIT_EMA400 | -77.10 |
| SELL | 2024-07-29 12:15:00 | 3089.90 | 2024-07-30 11:15:00 | 3119.60 | EXIT_EMA400 | -29.70 |
| SELL | 2024-10-30 15:15:00 | 2959.00 | 2024-11-06 11:15:00 | 3022.65 | EXIT_EMA400 | -63.65 |
| BUY | 2025-06-04 10:15:00 | 2478.00 | 2025-06-10 11:15:00 | 2650.65 | TARGET | 172.65 |
| SELL | 2026-01-06 10:15:00 | 2265.00 | 2026-01-20 13:15:00 | 2074.45 | TARGET | 190.55 |
