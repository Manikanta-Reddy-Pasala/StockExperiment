# Mankind Pharma Ltd. (MANKIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 2246.70
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 1 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / EMA400 exits:** 2 / 4
- **Total realized P&L (per unit):** -134.03
- **Avg P&L per closed trade:** -22.34

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 2104.55 | 2213.26 | 2213.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 2087.30 | 2193.11 | 2202.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 2183.60 | 2177.96 | 2193.72 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-05 14:15:00 | 2153.55 | 2177.57 | 2193.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 2162.65 | 2174.13 | 2190.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-06-12 11:15:00 | 2190.00 | 2174.27 | 2189.06 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 2297.65 | 2148.47 | 2148.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 2323.05 | 2157.14 | 2152.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 2586.00 | 2589.15 | 2481.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-30 09:15:00 | 2639.65 | 2557.91 | 2483.74 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-13 09:15:00 | 2514.95 | 2614.03 | 2536.39 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 2457.65 | 2657.33 | 2657.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 15:15:00 | 2440.00 | 2655.16 | 2656.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 2337.50 | 2309.28 | 2411.09 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 12:15:00 | 2555.00 | 2437.49 | 2437.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 2591.00 | 2442.57 | 2439.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 2457.10 | 2460.97 | 2449.70 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2490.00 | 2442.17 | 2441.32 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-22 09:15:00 | 2443.50 | 2484.39 | 2465.48 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 11:15:00 | 2358.20 | 2453.96 | 2454.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 2355.00 | 2452.05 | 2453.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 2436.40 | 2433.59 | 2443.16 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-11 11:15:00 | 2405.70 | 2433.32 | 2442.98 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-07-02 12:15:00 | 2408.30 | 2363.63 | 2396.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 2667.50 | 2418.91 | 2418.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 2695.00 | 2421.66 | 2419.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 2521.80 | 2545.81 | 2503.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-21 10:15:00 | 2606.00 | 2516.05 | 2497.13 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 14:15:00 | 2500.70 | 2528.79 | 2506.23 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 2465.10 | 2520.69 | 2520.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2442.20 | 2508.87 | 2514.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 2494.30 | 2490.54 | 2503.49 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-20 13:15:00 | 2480.90 | 2490.48 | 2503.26 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-07 09:15:00 | 2270.00 | 2203.92 | 2259.43 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 2277.10 | 2129.10 | 2128.45 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-06-05 14:15:00 | 2153.55 | 2024-06-12 11:15:00 | 2190.00 | EXIT_EMA400 | -36.45 |
| BUY | 2024-10-30 09:15:00 | 2639.65 | 2024-11-13 09:15:00 | 2514.95 | EXIT_EMA400 | -124.70 |
| BUY | 2025-05-12 09:15:00 | 2490.00 | 2025-05-22 09:15:00 | 2443.50 | EXIT_EMA400 | -46.50 |
| SELL | 2025-06-11 11:15:00 | 2405.70 | 2025-06-20 12:15:00 | 2293.87 | TARGET | 111.83 |
| BUY | 2025-08-21 10:15:00 | 2606.00 | 2025-08-26 14:15:00 | 2500.70 | EXIT_EMA400 | -105.30 |
| SELL | 2025-10-20 13:15:00 | 2480.90 | 2025-10-27 15:15:00 | 2413.81 | TARGET | 67.09 |
