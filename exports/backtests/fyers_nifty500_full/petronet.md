# Petronet LNG Ltd. (PETRONET.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 279.00
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 3 |
| ENTRY1 | 5 |
| ENTRY2 | 2 |
| EXIT | 5 |

## P&L

- **Trades closed:** 7
- **Trades open at end:** 0
- **Winners / losers:** 3 / 4
- **Target hits / EMA400 exits:** 3 / 4
- **Total realized P&L (per unit):** 36.80
- **Avg P&L per closed trade:** 5.26

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 329.90 | 345.23 | 345.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 326.45 | 344.90 | 345.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 344.60 | 343.67 | 344.39 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2024-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 09:15:00 | 361.90 | 345.12 | 345.09 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 326.70 | 346.22 | 346.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 09:15:00 | 322.35 | 340.40 | 342.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 10:15:00 | 332.60 | 331.87 | 337.52 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-26 13:15:00 | 330.70 | 332.02 | 337.32 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-29 15:15:00 | 337.00 | 331.20 | 336.30 | Close above EMA400 |

### Cycle 4 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 346.55 | 338.08 | 338.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 348.25 | 338.18 | 338.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 326.70 | 338.35 | 338.19 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 12:15:00 | 323.10 | 337.90 | 337.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 321.85 | 334.89 | 336.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 12:15:00 | 329.95 | 329.86 | 333.00 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-24 09:15:00 | 323.85 | 329.82 | 332.91 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-21 13:15:00 | 304.05 | 293.75 | 303.35 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 315.00 | 302.87 | 302.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 315.40 | 304.69 | 303.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 312.30 | 312.34 | 308.74 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-29 11:15:00 | 313.75 | 312.33 | 308.79 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-05-30 10:15:00 | 308.80 | 312.31 | 308.88 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 295.30 | 307.16 | 307.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 292.00 | 306.78 | 306.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 305.00 | 303.70 | 305.25 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-06-30 12:15:00 | 300.65 | 303.60 | 305.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 302.55 | 302.75 | 304.44 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-07-07 10:15:00 | 307.40 | 302.80 | 304.46 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 288.35 | 277.23 | 277.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 11:15:00 | 293.30 | 278.24 | 277.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 282.00 | 282.29 | 280.20 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2026-01-28 09:15:00 | 288.25 | 280.47 | 279.58 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 288.25 | 280.47 | 279.58 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-01-28 10:15:00 | 291.65 | 280.58 | 279.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 285.10 | 282.66 | 280.85 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2026-02-02 12:15:00 | 286.25 | 282.73 | 280.95 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2026-03-04 09:15:00 | 277.35 | 299.69 | 292.60 | Close below EMA400 |

### Cycle 9 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 237.00 | 288.95 | 288.96 | EMA200 below EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-11-26 13:15:00 | 330.70 | 2024-11-29 15:15:00 | 337.00 | EXIT_EMA400 | -6.30 |
| SELL | 2025-01-24 09:15:00 | 323.85 | 2025-01-28 13:15:00 | 296.66 | TARGET | 27.19 |
| BUY | 2025-05-29 11:15:00 | 313.75 | 2025-05-30 10:15:00 | 308.80 | EXIT_EMA400 | -4.95 |
| SELL | 2025-06-30 12:15:00 | 300.65 | 2025-07-07 10:15:00 | 307.40 | EXIT_EMA400 | -6.75 |
| BUY | 2026-02-02 12:15:00 | 286.25 | 2026-02-04 09:15:00 | 302.16 | TARGET | 15.91 |
| BUY | 2026-01-28 09:15:00 | 288.25 | 2026-02-25 10:15:00 | 314.25 | TARGET | 26.00 |
| BUY | 2026-01-28 10:15:00 | 291.65 | 2026-03-04 09:15:00 | 277.35 | EXIT_EMA400 | -14.30 |
