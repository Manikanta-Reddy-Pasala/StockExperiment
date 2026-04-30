# Mankind Pharma Ltd. (MANKIND.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2244.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT3 | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** -214.01
- **Avg P&L per closed trade:** -53.50

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 2306.60 | 2146.47 | 2146.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 14:15:00 | 2358.30 | 2176.82 | 2161.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 11:15:00 | 2586.00 | 2589.10 | 2480.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-10-30 09:15:00 | 2643.65 | 2557.84 | 2483.49 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-11-13 09:15:00 | 2514.80 | 2615.70 | 2537.87 | Close below EMA400 |

### Cycle 2 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 2457.55 | 2657.36 | 2657.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 2441.60 | 2628.09 | 2642.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 2337.50 | 2308.13 | 2409.35 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 11:15:00 | 2571.30 | 2435.79 | 2435.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 2591.00 | 2441.87 | 2438.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 2457.00 | 2460.42 | 2448.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-12 09:15:00 | 2490.00 | 2441.53 | 2440.40 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-22 09:15:00 | 2443.50 | 2484.02 | 2464.79 | Close below EMA400 |

### Cycle 4 — SELL (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 12:15:00 | 2358.60 | 2452.60 | 2453.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 2355.00 | 2451.63 | 2452.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 2363.40 | 2362.98 | 2395.92 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 2667.50 | 2418.81 | 2417.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 13:15:00 | 2694.50 | 2421.55 | 2419.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 10:15:00 | 2521.80 | 2545.65 | 2502.89 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-08-21 10:15:00 | 2606.00 | 2516.02 | 2497.00 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-08-26 14:15:00 | 2500.20 | 2528.76 | 2506.12 | Close below EMA400 |

### Cycle 6 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 2465.10 | 2520.80 | 2520.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 2442.20 | 2509.00 | 2514.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 2494.30 | 2490.63 | 2503.51 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-20 13:15:00 | 2480.90 | 2490.56 | 2503.28 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-01-07 09:15:00 | 2270.00 | 2203.89 | 2259.43 | Close above EMA400 |

### Cycle 7 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 2306.00 | 2127.83 | 2127.07 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-10-30 09:15:00 | 2643.65 | 2024-11-13 09:15:00 | 2514.80 | EXIT_EMA400 | -128.85 |
| BUY | 2025-05-12 09:15:00 | 2490.00 | 2025-05-22 09:15:00 | 2443.50 | EXIT_EMA400 | -46.50 |
| BUY | 2025-08-21 10:15:00 | 2606.00 | 2025-08-26 14:15:00 | 2500.20 | EXIT_EMA400 | -105.80 |
| SELL | 2025-10-20 13:15:00 | 2480.90 | 2025-10-27 15:15:00 | 2413.76 | TARGET | 67.14 |
