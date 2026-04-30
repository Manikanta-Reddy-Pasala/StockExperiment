# Bharat Heavy Electricals Ltd. (BHEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 352.41
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 1 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 22.57
- **Avg P&L per closed trade:** 5.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 09:15:00 | 274.00 | 296.31 | 296.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 266.80 | 294.74 | 295.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 10:15:00 | 278.80 | 277.89 | 285.02 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 11:15:00 | 271.35 | 279.11 | 284.33 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 252.85 | 241.31 | 252.57 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 10:15:00 | 227.98 | 209.79 | 209.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 11:15:00 | 228.30 | 209.97 | 209.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 218.14 | 219.60 | 215.66 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 222.32 | 219.59 | 215.72 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 216.48 | 219.93 | 216.12 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 214.79 | 219.87 | 216.11 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 227.97 | 246.36 | 246.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 224.65 | 246.14 | 246.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 225.07 | 222.73 | 230.45 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 10:15:00 | 240.70 | 233.61 | 233.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 243.43 | 235.06 | 234.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 273.50 | 274.40 | 262.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 275.60 | 274.20 | 262.67 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-08 14:15:00 | 272.05 | 284.75 | 275.13 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 253.65 | 269.48 | 269.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 250.70 | 267.55 | 268.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 267.65 | 266.89 | 268.13 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 09:15:00 | 261.15 | 268.02 | 268.51 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2026-02-27 09:15:00 | 266.45 | 263.50 | 265.60 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 294.43 | 262.71 | 262.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 310.53 | 265.84 | 264.26 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-03 11:15:00 | 271.35 | 2024-10-22 14:15:00 | 232.40 | TARGET | 38.95 |
| BUY | 2025-05-07 10:15:00 | 222.32 | 2025-05-09 09:15:00 | 214.79 | EXIT_EMA400 | -7.53 |
| BUY | 2025-12-09 10:15:00 | 275.60 | 2026-01-08 14:15:00 | 272.05 | EXIT_EMA400 | -3.55 |
| SELL | 2026-02-11 09:15:00 | 261.15 | 2026-02-27 09:15:00 | 266.45 | EXIT_EMA400 | -5.30 |
