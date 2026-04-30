# Asian Paints Ltd. (ASIANPAINT.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2448.80
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 6 |
| ENTRY1 | 5 |
| ENTRY2 | 4 |
| EXIT | 5 |

## P&L

- **Trades closed:** 9
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / EMA400 exits:** 2 / 7
- **Total realized P&L (per unit):** -192.10
- **Avg P&L per closed trade:** -21.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 11:15:00 | 2959.85 | 3114.88 | 3115.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 12:15:00 | 2948.80 | 3075.42 | 3093.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 14:15:00 | 2301.20 | 2298.91 | 2414.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-02-04 09:15:00 | 2274.20 | 2301.41 | 2406.64 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 13:15:00 | 2396.00 | 2302.55 | 2405.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-02-05 09:15:00 | 2262.65 | 2303.15 | 2403.90 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 2287.10 | 2248.09 | 2298.31 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-21 10:15:00 | 2300.00 | 2251.17 | 2297.91 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 10:15:00 | 2436.50 | 2321.02 | 2320.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 11:15:00 | 2442.20 | 2322.22 | 2321.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 2377.60 | 2391.17 | 2364.13 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 11:15:00 | 2308.60 | 2348.04 | 2348.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 12:15:00 | 2307.90 | 2347.65 | 2347.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 11:15:00 | 2348.00 | 2346.80 | 2347.46 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-27 13:15:00 | 2329.60 | 2346.58 | 2347.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 2329.60 | 2346.58 | 2347.34 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-05-27 14:15:00 | 2326.50 | 2346.38 | 2347.24 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2295.00 | 2278.59 | 2301.92 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-06-24 12:15:00 | 2282.00 | 2278.86 | 2301.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-06-27 09:15:00 | 2317.60 | 2279.83 | 2300.24 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 15:15:00 | 2428.00 | 2316.53 | 2316.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 09:15:00 | 2448.00 | 2317.84 | 2316.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-21 12:15:00 | 2369.70 | 2371.85 | 2350.18 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-21 15:15:00 | 2378.80 | 2371.91 | 2350.53 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 11:15:00 | 2344.00 | 2370.73 | 2351.66 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 2346.40 | 2446.74 | 2447.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 2336.00 | 2435.28 | 2441.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 14:15:00 | 2408.40 | 2404.32 | 2422.56 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-10-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 15:15:00 | 2514.60 | 2438.15 | 2437.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 09:15:00 | 2524.60 | 2443.44 | 2440.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-09 14:15:00 | 2794.50 | 2797.69 | 2683.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-10 09:15:00 | 2819.20 | 2797.87 | 2684.70 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2756.00 | 2805.32 | 2754.25 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-19 09:15:00 | 2765.90 | 2804.45 | 2754.32 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 2764.10 | 2803.51 | 2754.59 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2026-01-19 14:15:00 | 2753.30 | 2802.68 | 2754.66 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 2431.80 | 2720.57 | 2721.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 11:15:00 | 2374.00 | 2700.46 | 2710.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 2280.80 | 2263.38 | 2367.23 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-04-08 10:15:00 | 2269.10 | 2263.44 | 2366.74 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-04-10 09:15:00 | 2364.60 | 2265.40 | 2361.24 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-02-04 09:15:00 | 2274.20 | 2025-03-21 10:15:00 | 2300.00 | EXIT_EMA400 | -25.80 |
| SELL | 2025-02-05 09:15:00 | 2262.65 | 2025-03-21 10:15:00 | 2300.00 | EXIT_EMA400 | -37.35 |
| SELL | 2025-05-27 13:15:00 | 2329.60 | 2025-05-30 10:15:00 | 2276.37 | TARGET | 53.23 |
| SELL | 2025-05-27 14:15:00 | 2326.50 | 2025-05-30 12:15:00 | 2264.28 | TARGET | 62.22 |
| SELL | 2025-06-24 12:15:00 | 2282.00 | 2025-06-27 09:15:00 | 2317.60 | EXIT_EMA400 | -35.60 |
| BUY | 2025-07-21 15:15:00 | 2378.80 | 2025-07-24 11:15:00 | 2344.00 | EXIT_EMA400 | -34.80 |
| BUY | 2025-12-10 09:15:00 | 2819.20 | 2026-01-19 14:15:00 | 2753.30 | EXIT_EMA400 | -65.90 |
| BUY | 2026-01-19 09:15:00 | 2765.90 | 2026-01-19 14:15:00 | 2753.30 | EXIT_EMA400 | -12.60 |
| SELL | 2026-04-08 10:15:00 | 2269.10 | 2026-04-10 09:15:00 | 2364.60 | EXIT_EMA400 | -95.50 |
