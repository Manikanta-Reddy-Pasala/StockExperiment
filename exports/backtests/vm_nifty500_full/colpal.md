# Colgate Palmolive (India) Ltd. (COLPAL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 2096.20
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
- **Total realized P&L (per unit):** 304.78
- **Avg P&L per closed trade:** 76.20

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 13:15:00 | 3081.40 | 3458.98 | 3459.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 3063.95 | 3424.60 | 3441.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 3053.90 | 3024.08 | 3178.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-28 10:15:00 | 3027.95 | 3024.12 | 3177.34 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2930.90 | 2812.81 | 2932.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-01-10 09:15:00 | 2849.05 | 2817.13 | 2930.71 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-01-31 11:15:00 | 2853.20 | 2759.04 | 2846.79 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 2614.10 | 2566.28 | 2566.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 2642.10 | 2567.04 | 2566.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 09:15:00 | 2510.20 | 2594.35 | 2581.45 | EMA200 retest candle locked |

### Cycle 3 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 2499.50 | 2569.96 | 2570.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 2487.40 | 2565.36 | 2567.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2441.00 | 2440.72 | 2481.01 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-10 10:15:00 | 2421.00 | 2443.08 | 2475.69 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 2337.60 | 2276.06 | 2342.77 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-20 11:15:00 | 2348.50 | 2276.78 | 2342.80 | Close above EMA400 |

### Cycle 4 — BUY (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 13:15:00 | 2224.70 | 2148.03 | 2148.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 2237.20 | 2148.92 | 2148.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 10:15:00 | 2171.70 | 2176.27 | 2163.78 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-03-05 12:15:00 | 2188.50 | 2176.45 | 2163.99 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-03-09 09:15:00 | 2158.60 | 2178.76 | 2165.85 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1977.80 | 2153.95 | 2154.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1966.40 | 2152.09 | 2153.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 10:15:00 | 1962.50 | 1956.35 | 2022.31 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-28 10:15:00 | 3027.95 | 2025-01-31 11:15:00 | 2853.20 | EXIT_EMA400 | 174.75 |
| SELL | 2025-01-10 09:15:00 | 2849.05 | 2025-01-31 11:15:00 | 2853.20 | EXIT_EMA400 | -4.15 |
| SELL | 2025-07-10 10:15:00 | 2421.00 | 2025-07-24 11:15:00 | 2256.92 | TARGET | 164.08 |
| BUY | 2026-03-05 12:15:00 | 2188.50 | 2026-03-09 09:15:00 | 2158.60 | EXIT_EMA400 | -29.90 |
