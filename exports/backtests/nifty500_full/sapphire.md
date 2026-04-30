# Sapphire Foods India Ltd. (SAPPHIRE.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5004 bars)
- **Last close:** 205.61
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 18 |
| ALERT1 | 15 |
| ALERT2 | 15 |
| ALERT3 | 6 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 10 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 4 / 8
- **Target hits / EMA400 exits:** 4 / 8
- **Total realized P&L (per unit):** 12.31
- **Avg P&L per closed trade:** 1.03

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 13:15:00 | 291.87 | 276.83 | 276.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 10:15:00 | 292.88 | 277.40 | 277.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-14 11:15:00 | 283.46 | 283.56 | 280.59 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-09-14 14:15:00 | 285.00 | 283.56 | 280.63 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 283.58 | 286.91 | 283.37 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2023-09-28 14:15:00 | 282.93 | 286.85 | 283.37 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 13:15:00 | 261.86 | 282.53 | 282.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-27 14:15:00 | 258.59 | 281.18 | 281.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 10:15:00 | 272.38 | 270.65 | 275.29 | EMA200 retest candle locked |

### Cycle 3 — BUY (started 2023-12-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 11:15:00 | 286.06 | 277.50 | 277.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 13:15:00 | 286.77 | 277.68 | 277.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 277.00 | 280.02 | 278.99 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 15:15:00 | 285.00 | 280.22 | 279.17 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-12-28 10:15:00 | 279.36 | 280.83 | 279.57 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 274.80 | 281.43 | 281.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 11:15:00 | 272.56 | 281.35 | 281.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 280.81 | 279.12 | 280.21 | EMA200 retest candle locked |

### Cycle 5 — BUY (started 2024-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 13:15:00 | 303.91 | 281.30 | 281.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-07 10:15:00 | 305.00 | 289.96 | 286.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 11:15:00 | 289.71 | 291.17 | 287.12 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-18 10:15:00 | 293.17 | 288.60 | 286.26 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 300.61 | 302.86 | 296.64 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-04-18 11:15:00 | 296.49 | 302.49 | 296.93 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 11:15:00 | 281.49 | 293.20 | 293.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 09:15:00 | 279.90 | 292.62 | 292.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 15:15:00 | 285.00 | 284.35 | 287.86 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-05-22 10:15:00 | 282.42 | 284.32 | 287.81 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2024-05-29 09:15:00 | 288.00 | 283.63 | 286.90 | Close above EMA400 |

### Cycle 7 — BUY (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 12:15:00 | 302.16 | 288.64 | 288.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 12:15:00 | 302.98 | 290.24 | 289.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 10:15:00 | 305.60 | 306.48 | 300.06 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-07-10 11:15:00 | 308.06 | 306.49 | 300.10 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-07-22 09:15:00 | 301.15 | 307.51 | 302.04 | Close below EMA400 |

### Cycle 8 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 317.05 | 333.90 | 333.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 312.80 | 333.00 | 333.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 321.65 | 320.53 | 326.06 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-13 09:15:00 | 312.50 | 327.14 | 328.16 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 324.40 | 321.77 | 324.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-12-23 13:15:00 | 319.00 | 321.74 | 324.94 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-12-24 09:15:00 | 326.15 | 321.74 | 324.89 | Close above EMA400 |

### Cycle 9 — BUY (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 13:15:00 | 345.65 | 327.02 | 326.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 09:15:00 | 365.75 | 327.82 | 327.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 13:15:00 | 329.15 | 332.58 | 330.09 | EMA200 retest candle locked |

### Cycle 10 — SELL (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 11:15:00 | 313.90 | 328.10 | 328.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 13:15:00 | 310.35 | 326.81 | 327.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 09:15:00 | 322.60 | 314.22 | 320.33 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-26 09:15:00 | 290.30 | 310.26 | 313.54 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-04-15 09:15:00 | 310.35 | 302.56 | 307.88 | Close above EMA400 |

### Cycle 11 — BUY (started 2025-04-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 13:15:00 | 318.40 | 311.34 | 311.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 323.80 | 311.64 | 311.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 309.10 | 313.10 | 312.28 | EMA200 retest candle locked |

### Cycle 12 — SELL (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 15:15:00 | 300.10 | 311.49 | 311.52 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 335.65 | 311.51 | 311.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 09:15:00 | 349.40 | 323.01 | 319.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 326.60 | 327.22 | 322.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-07-11 12:15:00 | 329.20 | 327.24 | 322.74 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 330.90 | 331.19 | 326.07 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2025-07-23 14:15:00 | 342.95 | 331.30 | 326.16 | Buy entry 2 (retest2 break) |
| Exit — 1H close below EMA400 | 2025-07-28 09:15:00 | 326.55 | 331.89 | 326.87 | Close below EMA400 |

### Cycle 14 — SELL (started 2025-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 09:15:00 | 318.05 | 323.87 | 323.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 09:15:00 | 310.00 | 323.26 | 323.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 13:15:00 | 322.65 | 321.42 | 322.58 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-28 11:15:00 | 320.30 | 322.96 | 323.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 319.45 | 322.82 | 323.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-08-29 10:15:00 | 325.15 | 322.83 | 323.15 | Close above EMA400 |

### Cycle 15 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 325.35 | 323.41 | 323.41 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 322.00 | 323.40 | 323.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 12:15:00 | 321.10 | 323.36 | 323.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 333.15 | 323.00 | 323.19 | EMA200 retest candle locked |

### Cycle 17 — BUY (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 13:15:00 | 333.10 | 323.39 | 323.38 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 312.70 | 323.63 | 323.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 311.90 | 323.52 | 323.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 09:15:00 | 297.65 | 294.68 | 303.98 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-10-31 10:15:00 | 290.70 | 295.06 | 303.81 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 254.40 | 247.60 | 260.15 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-01-01 14:15:00 | 261.95 | 248.06 | 260.07 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-09-14 14:15:00 | 285.00 | 2023-09-15 14:15:00 | 298.10 | TARGET | 13.10 |
| BUY | 2023-12-22 15:15:00 | 285.00 | 2023-12-28 10:15:00 | 279.36 | EXIT_EMA400 | -5.64 |
| BUY | 2024-03-18 10:15:00 | 293.17 | 2024-03-20 09:15:00 | 313.89 | TARGET | 20.72 |
| SELL | 2024-05-22 10:15:00 | 282.42 | 2024-05-29 09:15:00 | 288.00 | EXIT_EMA400 | -5.58 |
| BUY | 2024-07-10 11:15:00 | 308.06 | 2024-07-22 09:15:00 | 301.15 | EXIT_EMA400 | -6.91 |
| SELL | 2024-12-13 09:15:00 | 312.50 | 2024-12-24 09:15:00 | 326.15 | EXIT_EMA400 | -13.65 |
| SELL | 2024-12-23 13:15:00 | 319.00 | 2024-12-24 09:15:00 | 326.15 | EXIT_EMA400 | -7.15 |
| SELL | 2025-03-26 09:15:00 | 290.30 | 2025-04-15 09:15:00 | 310.35 | EXIT_EMA400 | -20.05 |
| BUY | 2025-07-11 12:15:00 | 329.20 | 2025-07-17 12:15:00 | 348.59 | TARGET | 19.39 |
| BUY | 2025-07-23 14:15:00 | 342.95 | 2025-07-28 09:15:00 | 326.55 | EXIT_EMA400 | -16.40 |
| SELL | 2025-08-28 11:15:00 | 320.30 | 2025-08-29 10:15:00 | 325.15 | EXIT_EMA400 | -4.85 |
| SELL | 2025-10-31 10:15:00 | 290.70 | 2025-11-13 09:15:00 | 251.37 | TARGET | 39.33 |
