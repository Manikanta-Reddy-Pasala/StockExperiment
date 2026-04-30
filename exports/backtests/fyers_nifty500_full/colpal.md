# Colgate Palmolive (India) Ltd. (COLPAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 2110.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 1 |
| EXIT | 3 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 303.70
- **Avg P&L per closed trade:** 75.92

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 12:15:00 | 3069.55 | 3463.00 | 3463.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 3046.10 | 3417.42 | 3440.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 3053.90 | 3021.92 | 3177.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 3028.00 | 3021.98 | 3176.46 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2930.90 | 2812.50 | 2931.72 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-10 09:15:00 | 2848.40 | 2816.85 | 2930.41 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 2853.20 | 2758.67 | 2846.45 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 2615.00 | 2566.36 | 2566.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2642.10 | 2567.11 | 2566.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 2510.20 | 2594.45 | 2581.60 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 2499.50 | 2570.04 | 2570.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 2487.40 | 2565.44 | 2567.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2441.00 | 2440.75 | 2481.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 10:15:00 | 2421.00 | 2443.09 | 2475.73 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2337.60 | 2276.04 | 2342.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-20 11:15:00 | 2348.50 | 2276.76 | 2342.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 12:15:00 | 2216.00 | 2146.91 | 2146.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 2223.70 | 2147.67 | 2147.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 2171.70 | 2176.15 | 2163.22 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 12:15:00 | 2188.50 | 2176.33 | 2163.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 2158.00 | 2178.65 | 2165.32 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1977.80 | 2153.87 | 2154.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1966.30 | 2152.00 | 2153.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 1962.50 | 1956.38 | 2022.12 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 10:15:00 | 3028.00 | 2025-01-31 11:15:00 | 2853.20 | EXIT_EMA400 | 174.80 |
| SELL | 2025-01-10 09:15:00 | 2848.40 | 2025-01-31 11:15:00 | 2853.20 | EXIT_EMA400 | -4.80 |
| SELL | 2025-07-10 10:15:00 | 2421.00 | 2025-07-24 11:15:00 | 2256.80 | TARGET | 164.20 |
| BUY | 2026-03-05 12:15:00 | 2188.50 | 2026-03-09 09:15:00 | 2158.00 | EXIT_EMA400 | -30.50 |
