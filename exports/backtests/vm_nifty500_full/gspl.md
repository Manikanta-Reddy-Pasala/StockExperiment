# Gujarat State Petronet Ltd. (GSPL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:30:00 (5004 bars)
- **Last close:** 285.19
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 14 |
| ALERT1 | 12 |
| ALERT2 | 12 |
| ALERT3 | 5 |
| ENTRY1 | 10 |
| ENTRY2 | 2 |
| EXIT | 10 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 3 / 9
- **Target hits / EMA400 exits:** 3 / 9
- **Total realized P&L (per unit):** -2.83
- **Avg P&L per closed trade:** -0.24

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-10-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 14:15:00 | 289.00 | 285.44 | 285.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 15:15:00 | 289.40 | 285.48 | 285.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-19 15:15:00 | 285.30 | 285.54 | 285.48 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-10-20 09:15:00 | 288.65 | 285.57 | 285.50 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2023-10-20 12:15:00 | 285.00 | 285.60 | 285.51 | Close below EMA400 |

### Cycle 2 — SELL (started 2023-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 12:15:00 | 277.65 | 285.41 | 285.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 14:15:00 | 272.80 | 285.21 | 285.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 283.45 | 277.89 | 281.11 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2023-11-07 11:15:00 | 279.60 | 277.96 | 281.12 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 279.60 | 277.96 | 281.12 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2023-11-07 13:15:00 | 277.15 | 277.97 | 281.09 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2023-11-15 09:15:00 | 280.90 | 277.35 | 280.30 | Close above EMA400 |

### Cycle 3 — BUY (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 12:15:00 | 297.35 | 281.13 | 281.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 09:15:00 | 298.35 | 281.80 | 281.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 13:15:00 | 282.70 | 286.02 | 283.97 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2023-12-22 09:15:00 | 294.50 | 286.24 | 284.18 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-13 09:15:00 | 345.80 | 360.48 | 346.54 | Close below EMA400 |

### Cycle 4 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 293.15 | 349.13 | 349.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 14:15:00 | 292.00 | 333.53 | 340.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-12 10:15:00 | 303.90 | 297.38 | 311.17 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-06-27 09:15:00 | 295.50 | 300.29 | 308.89 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 307.65 | 300.91 | 307.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-07-05 09:15:00 | 312.00 | 301.14 | 307.69 | Close above EMA400 |

### Cycle 5 — BUY (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 10:15:00 | 330.70 | 311.49 | 311.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 333.00 | 313.18 | 312.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 323.40 | 326.22 | 320.40 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-08-21 12:15:00 | 329.05 | 324.89 | 320.81 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-10-18 09:15:00 | 388.65 | 406.85 | 390.55 | Close below EMA400 |

### Cycle 6 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 328.85 | 385.37 | 385.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 325.50 | 384.78 | 385.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 15:15:00 | 367.95 | 364.54 | 372.88 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-12-23 12:15:00 | 361.85 | 370.70 | 373.65 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-01-03 09:15:00 | 371.10 | 367.01 | 370.86 | Close above EMA400 |

### Cycle 7 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 320.65 | 310.68 | 310.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 15:15:00 | 325.25 | 311.34 | 311.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 13:15:00 | 328.55 | 328.74 | 321.65 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-28 09:15:00 | 329.80 | 328.76 | 321.77 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 327.25 | 330.63 | 325.39 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-06-18 15:15:00 | 325.65 | 330.21 | 325.67 | Close below EMA400 |

### Cycle 8 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 303.20 | 325.94 | 326.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 299.15 | 321.52 | 323.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 309.00 | 304.95 | 311.24 | EMA200 retest candle locked |

### Cycle 9 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 320.35 | 313.72 | 313.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 320.75 | 314.18 | 313.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 09:15:00 | 314.35 | 314.44 | 314.07 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-10-15 13:15:00 | 319.95 | 314.57 | 314.14 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 314.90 | 314.96 | 314.36 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-10-17 11:15:00 | 314.25 | 314.95 | 314.36 | Close below EMA400 |

### Cycle 10 — SELL (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 15:15:00 | 312.45 | 313.91 | 313.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 310.90 | 313.88 | 313.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 307.00 | 301.59 | 306.10 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-12-08 14:15:00 | 290.50 | 300.28 | 304.73 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-12-31 09:15:00 | 301.75 | 293.74 | 298.48 | Close above EMA400 |

### Cycle 11 — BUY (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 09:15:00 | 311.85 | 302.20 | 302.17 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2026-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 13:15:00 | 294.40 | 302.25 | 302.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 09:15:00 | 289.35 | 301.70 | 301.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 09:15:00 | 305.65 | 301.16 | 301.67 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-02-02 09:15:00 | 295.25 | 301.46 | 301.79 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 301.60 | 301.19 | 301.64 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-02-03 10:15:00 | 299.55 | 301.17 | 301.63 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2026-02-03 11:15:00 | 301.80 | 301.18 | 301.63 | Close above EMA400 |

### Cycle 13 — BUY (started 2026-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 09:15:00 | 314.55 | 301.95 | 301.95 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 13:15:00 | 286.15 | 302.74 | 302.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-05 13:15:00 | 283.05 | 301.61 | 302.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 275.04 | 255.05 | 270.23 | EMA200 retest candle locked |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2023-10-20 09:15:00 | 288.65 | 2023-10-20 12:15:00 | 285.00 | EXIT_EMA400 | -3.65 |
| SELL | 2023-11-07 11:15:00 | 279.60 | 2023-11-08 10:15:00 | 275.05 | TARGET | 4.55 |
| SELL | 2023-11-07 13:15:00 | 277.15 | 2023-11-15 09:15:00 | 280.90 | EXIT_EMA400 | -3.75 |
| BUY | 2023-12-22 09:15:00 | 294.50 | 2024-01-02 09:15:00 | 325.46 | TARGET | 30.96 |
| SELL | 2024-06-27 09:15:00 | 295.50 | 2024-07-05 09:15:00 | 312.00 | EXIT_EMA400 | -16.50 |
| BUY | 2024-08-21 12:15:00 | 329.05 | 2024-08-26 09:15:00 | 353.76 | TARGET | 24.71 |
| SELL | 2024-12-23 12:15:00 | 361.85 | 2025-01-03 09:15:00 | 371.10 | EXIT_EMA400 | -9.25 |
| BUY | 2025-05-28 09:15:00 | 329.80 | 2025-06-18 15:15:00 | 325.65 | EXIT_EMA400 | -4.15 |
| BUY | 2025-10-15 13:15:00 | 319.95 | 2025-10-17 11:15:00 | 314.25 | EXIT_EMA400 | -5.70 |
| SELL | 2025-12-08 14:15:00 | 290.50 | 2025-12-31 09:15:00 | 301.75 | EXIT_EMA400 | -11.25 |
| SELL | 2026-02-02 09:15:00 | 295.25 | 2026-02-03 11:15:00 | 301.80 | EXIT_EMA400 | -6.55 |
| SELL | 2026-02-03 10:15:00 | 299.55 | 2026-02-03 11:15:00 | 301.80 | EXIT_EMA400 | -2.25 |
