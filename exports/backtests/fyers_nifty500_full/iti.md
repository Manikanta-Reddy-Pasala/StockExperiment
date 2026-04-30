# ITI Ltd. (ITI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:15:00 (3409 bars)
- **Last close:** 300.95
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 2 |
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| EXIT | 3 |

## P&L

- **Trades closed:** 3
- **Trades open at end:** 0
- **Winners / losers:** 0 / 3
- **Target hits / EMA400 exits:** 0 / 3
- **Total realized P&L (per unit):** -26.90
- **Avg P&L per closed trade:** -8.97

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 279.75 | 298.42 | 298.43 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 306.75 | 297.96 | 297.96 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 281.90 | 297.92 | 297.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-20 10:15:00 | 277.65 | 293.28 | 295.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-07 09:15:00 | 254.23 | 240.15 | 256.89 | EMA200 retest candle locked |

### Cycle 4 — BUY (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 11:15:00 | 294.50 | 268.42 | 268.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 12:15:00 | 298.50 | 268.72 | 268.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 385.55 | 387.60 | 349.34 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 14:15:00 | 283.00 | 338.79 | 339.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-12 09:15:00 | 268.00 | 337.52 | 338.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 268.20 | 257.99 | 275.22 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-06 11:15:00 | 252.25 | 262.25 | 272.06 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 266.25 | 258.85 | 268.36 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 12:15:00 | 272.60 | 259.15 | 268.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 340.15 | 274.05 | 273.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 360.50 | 275.56 | 274.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 305.80 | 309.51 | 297.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 311.65 | 309.47 | 297.35 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 12:15:00 | 308.85 | 316.99 | 309.99 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 291.95 | 305.43 | 305.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 291.20 | 305.28 | 305.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 305.85 | 297.73 | 301.05 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 324.00 | 303.58 | 303.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 09:15:00 | 361.10 | 311.07 | 308.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 325.20 | 325.76 | 319.05 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 308.55 | 316.81 | 316.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 306.35 | 316.48 | 316.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 326.80 | 308.08 | 311.53 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-19 15:15:00 | 309.85 | 308.65 | 311.72 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 309.85 | 308.65 | 311.72 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 13:15:00 | 313.60 | 308.76 | 311.70 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 09:15:00 | 300.39 | 280.60 | 280.55 | EMA200 above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2025-05-06 11:15:00 | 252.25 | 2025-05-14 12:15:00 | 272.60 | EXIT_EMA400 | -20.35 |
| BUY | 2025-06-20 10:15:00 | 311.65 | 2025-07-24 12:15:00 | 308.85 | EXIT_EMA400 | -2.80 |
| SELL | 2025-12-19 15:15:00 | 309.85 | 2025-12-22 13:15:00 | 313.60 | EXIT_EMA400 | -3.75 |
