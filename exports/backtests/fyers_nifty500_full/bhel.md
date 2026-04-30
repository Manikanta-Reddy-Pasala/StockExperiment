# Bharat Heavy Electricals Ltd. (BHEL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 352.45
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT3 | 2 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| EXIT | 4 |

## P&L

- **Trades closed:** 4
- **Trades open at end:** 0
- **Winners / losers:** 1 / 3
- **Target hits / EMA400 exits:** 1 / 3
- **Total realized P&L (per unit):** 24.50
- **Avg P&L per closed trade:** 6.12

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 12:15:00 | 285.85 | 298.83 | 298.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 09:15:00 | 284.10 | 297.57 | 298.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 10:15:00 | 278.95 | 277.89 | 285.78 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-10-03 11:15:00 | 271.35 | 279.11 | 284.94 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-11-28 09:15:00 | 252.90 | 241.26 | 252.61 | Close above EMA400 |

### Cycle 2 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 225.20 | 209.57 | 209.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 09:15:00 | 230.35 | 211.93 | 210.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 14:15:00 | 218.14 | 219.56 | 215.56 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-07 10:15:00 | 222.32 | 219.55 | 215.61 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 216.48 | 219.89 | 216.01 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-05-09 09:15:00 | 214.79 | 219.84 | 216.01 | Close below EMA400 |

### Cycle 3 — SELL (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 15:15:00 | 228.18 | 246.38 | 246.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 09:15:00 | 224.65 | 246.16 | 246.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 225.13 | 222.74 | 230.46 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 243.26 | 233.55 | 233.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 246.13 | 235.44 | 234.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 273.50 | 274.40 | 262.36 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-12-09 10:15:00 | 275.60 | 274.20 | 262.68 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2026-01-08 14:15:00 | 272.05 | 284.76 | 275.14 | Close below EMA400 |

### Cycle 5 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 253.65 | 269.47 | 269.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 250.70 | 266.84 | 268.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 11:15:00 | 267.55 | 266.24 | 267.75 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-11 09:15:00 | 261.15 | 267.58 | 268.21 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 264.80 | 263.26 | 265.39 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-02-27 09:15:00 | 266.35 | 263.30 | 265.39 | Close above EMA400 |

### Cycle 6 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 294.43 | 262.68 | 262.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 14:15:00 | 310.49 | 265.81 | 264.18 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-10-03 11:15:00 | 271.35 | 2024-10-23 09:15:00 | 230.57 | TARGET | 40.78 |
| BUY | 2025-05-07 10:15:00 | 222.32 | 2025-05-09 09:15:00 | 214.79 | EXIT_EMA400 | -7.53 |
| BUY | 2025-12-09 10:15:00 | 275.60 | 2026-01-08 14:15:00 | 272.05 | EXIT_EMA400 | -3.55 |
| SELL | 2026-02-11 09:15:00 | 261.15 | 2026-02-27 09:15:00 | 266.35 | EXIT_EMA400 | -5.20 |
