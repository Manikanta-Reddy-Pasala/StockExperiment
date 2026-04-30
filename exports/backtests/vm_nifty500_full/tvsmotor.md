# TVS Motor Company Ltd. (TVSMOTOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 3492.90
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 6 |
| ALERT3 | 4 |
| ENTRY1 | 3 |
| ENTRY2 | 2 |
| EXIT | 3 |

## P&L

- **Trades closed:** 5
- **Trades open at end:** 0
- **Winners / losers:** 4 / 1
- **Target hits / EMA400 exits:** 3 / 2
- **Total realized P&L (per unit):** 684.61
- **Avg P&L per closed trade:** 136.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 13:15:00 | 1964.65 | 2049.62 | 2049.95 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 2133.90 | 2048.76 | 2048.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 2150.95 | 2058.53 | 2053.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 11:15:00 | 2107.00 | 2140.64 | 2103.47 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-06-04 12:15:00 | 2192.30 | 2141.16 | 2103.91 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 2192.30 | 2141.16 | 2103.91 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-04 13:15:00 | 2210.25 | 2141.85 | 2104.44 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2024-10-04 13:15:00 | 2673.45 | 2773.04 | 2676.57 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 2476.25 | 2656.85 | 2657.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2447.75 | 2649.91 | 2653.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 2558.25 | 2495.01 | 2549.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-05 09:15:00 | 2490.05 | 2499.99 | 2548.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 2534.55 | 2503.71 | 2543.09 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-12 09:15:00 | 2506.90 | 2504.05 | 2542.87 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 14:15:00 | 2500.15 | 2457.83 | 2500.25 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-01-02 15:15:00 | 2507.95 | 2458.33 | 2500.29 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 14:15:00 | 2482.25 | 2452.47 | 2452.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-13 09:15:00 | 2513.10 | 2454.80 | 2453.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-14 09:15:00 | 2440.70 | 2456.43 | 2454.44 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 09:15:00 | 2377.75 | 2452.40 | 2452.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 2352.00 | 2446.74 | 2449.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 2342.00 | 2340.53 | 2378.92 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2525.40 | 2401.94 | 2401.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2594.00 | 2409.14 | 2405.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 2726.30 | 2735.12 | 2655.61 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 14:15:00 | 2750.00 | 2735.00 | 2659.42 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2764.10 | 2826.37 | 2763.41 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 13:15:00 | 2762.00 | 2825.73 | 2763.40 | Close below EMA400 |

### Cycle 7 — SELL (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 11:15:00 | 3383.60 | 3685.00 | 3686.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3369.30 | 3595.39 | 3635.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3690.90 | 3550.15 | 3604.11 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 12:15:00 | 3705.60 | 3643.76 | 3643.63 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 3530.30 | 3643.50 | 3643.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 10:15:00 | 3525.50 | 3642.33 | 3642.93 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-06-04 12:15:00 | 2192.30 | 2024-06-13 13:15:00 | 2457.46 | TARGET | 265.16 |
| BUY | 2024-06-04 13:15:00 | 2210.25 | 2024-07-26 10:15:00 | 2527.67 | TARGET | 317.42 |
| SELL | 2024-12-12 09:15:00 | 2506.90 | 2024-12-20 14:15:00 | 2398.98 | TARGET | 107.92 |
| SELL | 2024-12-05 09:15:00 | 2490.05 | 2025-01-02 15:15:00 | 2507.95 | EXIT_EMA400 | -17.90 |
| BUY | 2025-06-06 14:15:00 | 2750.00 | 2025-07-11 13:15:00 | 2762.00 | EXIT_EMA400 | 12.00 |
