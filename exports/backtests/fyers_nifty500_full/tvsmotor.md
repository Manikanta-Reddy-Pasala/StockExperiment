# TVS Motor Company Ltd. (TVSMOTOR.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 3492.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT3 | 2 |
| ENTRY1 | 2 |
| ENTRY2 | 1 |
| EXIT | 2 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 2 / 1
- **Target hits / EMA400 exits:** 1 / 2
- **Total realized P&L (per unit):** 148.35
- **Avg P&L per closed trade:** 49.45

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 12:15:00 | 2477.25 | 2656.78 | 2657.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 2444.40 | 2647.07 | 2652.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 2494.60 | 2493.72 | 2548.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-17 11:15:00 | 2466.30 | 2505.15 | 2538.83 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-02 14:15:00 | 2500.15 | 2457.69 | 2499.85 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 10:15:00 | 2565.30 | 2451.25 | 2451.10 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 2350.75 | 2451.25 | 2451.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 2330.25 | 2436.05 | 2443.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 11:15:00 | 2342.00 | 2341.67 | 2380.00 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2525.40 | 2402.55 | 2402.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2594.00 | 2409.75 | 2406.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 11:15:00 | 2726.30 | 2735.14 | 2655.80 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-06 14:15:00 | 2750.00 | 2735.05 | 2659.62 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 2716.10 | 2739.52 | 2672.50 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-06-13 13:15:00 | 2730.10 | 2738.61 | 2673.37 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 2764.10 | 2826.23 | 2763.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-07-11 13:15:00 | 2762.00 | 2825.59 | 2763.36 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 10:15:00 | 3423.80 | 3687.62 | 3687.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-17 11:15:00 | 3383.60 | 3684.59 | 3686.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3690.90 | 3550.12 | 3604.21 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2026-04-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 14:15:00 | 3751.50 | 3644.27 | 3643.97 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 3494.10 | 3642.59 | 3643.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 3468.90 | 3634.31 | 3639.01 | Break + close below crossover candle low |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-12-17 11:15:00 | 2466.30 | 2025-01-02 14:15:00 | 2500.15 | EXIT_EMA400 | -33.85 |
| BUY | 2025-06-13 13:15:00 | 2730.10 | 2025-06-25 14:15:00 | 2900.30 | TARGET | 170.20 |
| BUY | 2025-06-06 14:15:00 | 2750.00 | 2025-07-11 13:15:00 | 2762.00 | EXIT_EMA400 | 12.00 |
