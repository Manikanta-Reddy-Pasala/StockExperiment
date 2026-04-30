# ITI Ltd. (ITI.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5005 bars)
- **Last close:** 298.88
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT3 | 3 |
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| EXIT | 6 |

## P&L

- **Trades closed:** 6
- **Trades open at end:** 0
- **Winners / losers:** 0 / 6
- **Target hits / EMA400 exits:** 0 / 6
- **Total realized P&L (per unit):** -62.25
- **Avg P&L per closed trade:** -10.38

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 15:15:00 | 260.85 | 303.22 | 303.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 249.75 | 302.69 | 303.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 13:15:00 | 276.15 | 275.50 | 285.21 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-04-08 10:15:00 | 273.20 | 275.47 | 285.00 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-04-22 12:15:00 | 279.10 | 268.41 | 278.69 | Close above EMA400 |

### Cycle 2 — BUY (started 2024-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 14:15:00 | 314.00 | 285.28 | 285.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 329.75 | 290.05 | 287.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 291.35 | 295.77 | 291.30 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-30 09:15:00 | 307.10 | 295.59 | 291.36 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 307.10 | 295.59 | 291.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-06-04 10:15:00 | 283.80 | 296.57 | 292.32 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 13:15:00 | 289.50 | 300.73 | 300.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 14:15:00 | 287.60 | 300.60 | 300.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-21 14:15:00 | 296.60 | 295.92 | 298.08 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-06 09:15:00 | 291.85 | 298.60 | 299.02 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-09-10 11:15:00 | 298.65 | 296.87 | 298.09 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 12:15:00 | 298.50 | 268.85 | 268.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 312.75 | 275.09 | 272.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-15 09:15:00 | 385.55 | 387.59 | 349.42 | EMA200 retest candle locked |

### Cycle 5 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 268.00 | 339.03 | 339.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 258.00 | 327.61 | 333.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-16 09:15:00 | 268.20 | 258.05 | 275.37 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-05-06 11:15:00 | 252.25 | 262.27 | 272.14 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 266.35 | 258.86 | 268.43 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-05-14 12:15:00 | 272.60 | 259.17 | 268.44 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 14:15:00 | 340.15 | 274.07 | 273.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 09:15:00 | 360.50 | 275.58 | 274.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 305.80 | 309.50 | 297.09 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-06-20 10:15:00 | 311.65 | 309.47 | 297.38 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-07-24 12:15:00 | 308.85 | 316.97 | 309.99 | Close below EMA400 |

### Cycle 7 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 291.95 | 305.42 | 305.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 291.20 | 305.27 | 305.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 09:15:00 | 305.85 | 297.75 | 301.06 | EMA200 retest candle locked |

### Cycle 8 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 324.00 | 303.60 | 303.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 09:15:00 | 361.10 | 311.07 | 308.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 325.20 | 325.78 | 319.07 | EMA200 retest candle locked |

### Cycle 9 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 308.55 | 316.81 | 316.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 09:15:00 | 306.35 | 316.48 | 316.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 326.80 | 308.09 | 311.54 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-22 09:15:00 | 310.50 | 308.71 | 311.74 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 310.50 | 308.71 | 311.74 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-12-22 13:15:00 | 313.60 | 308.81 | 311.73 | Close above EMA400 |

### Cycle 10 — BUY (started 2026-04-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-24 14:15:00 | 297.73 | 280.82 | 280.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 09:15:00 | 300.42 | 281.18 | 280.94 | Break + close above crossover candle high |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-04-08 10:15:00 | 273.20 | 2024-04-22 12:15:00 | 279.10 | EXIT_EMA400 | -5.90 |
| BUY | 2024-05-30 09:15:00 | 307.10 | 2024-06-04 10:15:00 | 283.80 | EXIT_EMA400 | -23.30 |
| SELL | 2024-09-06 09:15:00 | 291.85 | 2024-09-10 11:15:00 | 298.65 | EXIT_EMA400 | -6.80 |
| SELL | 2025-05-06 11:15:00 | 252.25 | 2025-05-14 12:15:00 | 272.60 | EXIT_EMA400 | -20.35 |
| BUY | 2025-06-20 10:15:00 | 311.65 | 2025-07-24 12:15:00 | 308.85 | EXIT_EMA400 | -2.80 |
| SELL | 2025-12-22 09:15:00 | 310.50 | 2025-12-22 13:15:00 | 313.60 | EXIT_EMA400 | -3.10 |
