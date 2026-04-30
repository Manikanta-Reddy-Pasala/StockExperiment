# Jupiter Wagons Ltd. (JWL.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 283.73
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT3 | 13 |
| ENTRY1 | 6 |
| ENTRY2 | 9 |
| EXIT | 6 |

## P&L

- **Trades closed:** 15
- **Trades open at end:** 0
- **Winners / losers:** 8 / 7
- **Target hits / EMA400 exits:** 7 / 8
- **Total realized P&L (per unit):** 249.67
- **Avg P&L per closed trade:** 16.64

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 14:15:00 | 355.45 | 356.40 | 356.40 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-26 09:15:00 | 360.95 | 356.43 | 356.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 10:15:00 | 367.70 | 356.55 | 356.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 365.75 | 371.31 | 365.31 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-04-15 10:15:00 | 367.95 | 371.28 | 365.33 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 15:15:00 | 366.20 | 371.15 | 365.41 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-16 09:15:00 | 370.40 | 371.14 | 365.43 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 371.50 | 371.75 | 366.13 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-04-19 13:15:00 | 373.60 | 371.73 | 366.23 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 499.95 | 503.41 | 455.21 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-06-05 14:15:00 | 540.00 | 504.65 | 457.02 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 601.15 | 660.60 | 600.81 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-22 09:15:00 | 639.00 | 657.85 | 601.46 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 609.00 | 655.93 | 603.23 | EMA400 retest candle locked |
| Second Entry (BUY) — break of retest2 high, sustain | 2024-07-24 09:15:00 | 642.70 | 654.67 | 603.64 | Buy entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 608.80 | 643.48 | 607.23 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2024-08-01 10:15:00 | 604.90 | 643.10 | 607.22 | Close below EMA400 |

### Cycle 3 — SELL (started 2024-08-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 12:15:00 | 557.20 | 587.91 | 587.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 15:15:00 | 554.00 | 586.94 | 587.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 12:15:00 | 558.55 | 557.83 | 569.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-09-16 09:15:00 | 546.80 | 557.79 | 569.58 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 504.45 | 503.22 | 522.96 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-04 14:15:00 | 500.35 | 503.44 | 522.58 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 518.90 | 503.72 | 521.24 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-11-07 14:15:00 | 510.80 | 504.24 | 521.08 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 496.00 | 474.86 | 497.62 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 11:15:00 | 498.45 | 477.19 | 497.08 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 524.35 | 507.12 | 507.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 14:15:00 | 535.70 | 507.56 | 507.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-26 14:15:00 | 510.75 | 511.57 | 509.54 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-01-03 09:15:00 | 520.80 | 508.92 | 508.43 | Buy entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 520.80 | 508.92 | 508.43 | EMA400 retest candle locked |
| Exit — 1H close below EMA400 | 2025-01-06 09:15:00 | 499.00 | 509.08 | 508.53 | Close below EMA400 |

### Cycle 5 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 477.50 | 507.76 | 507.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 476.50 | 505.14 | 506.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-16 11:15:00 | 487.40 | 486.31 | 495.99 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-01-22 10:15:00 | 466.30 | 487.16 | 495.24 | Sell entry 1 (retest1 break) |
| Exit — 1H close above EMA400 | 2025-03-24 09:15:00 | 365.90 | 324.76 | 363.38 | Close above EMA400 |

### Cycle 6 — BUY (started 2025-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 13:15:00 | 440.15 | 370.29 | 369.94 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 13:15:00 | 368.85 | 382.37 | 382.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 367.10 | 379.90 | 381.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 368.70 | 345.01 | 357.84 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-08-21 14:15:00 | 344.70 | 345.62 | 357.84 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 344.70 | 345.62 | 357.84 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-21 15:15:00 | 341.15 | 345.58 | 357.75 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-15 09:15:00 | 346.00 | 333.81 | 345.38 | Close above EMA400 |

### Cycle 8 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 325.50 | 309.05 | 308.97 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 293.00 | 308.93 | 308.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 290.25 | 308.61 | 308.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 11:15:00 | 326.20 | 306.88 | 307.86 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 09:15:00 | 326.40 | 308.96 | 308.87 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 299.80 | 309.52 | 309.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 293.80 | 308.88 | 309.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 10:15:00 | 287.95 | 287.67 | 296.63 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2026-03-10 09:15:00 | 280.55 | 288.08 | 296.28 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 276.95 | 268.38 | 280.33 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2026-04-07 13:15:00 | 271.68 | 268.47 | 280.25 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 275.16 | 268.46 | 278.58 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2026-04-16 10:15:00 | 282.59 | 268.91 | 278.41 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| BUY | 2024-04-15 10:15:00 | 367.95 | 2024-04-18 09:15:00 | 375.82 | TARGET | 7.87 |
| BUY | 2024-04-16 09:15:00 | 370.40 | 2024-04-22 10:15:00 | 385.30 | TARGET | 14.90 |
| BUY | 2024-04-19 13:15:00 | 373.60 | 2024-04-22 12:15:00 | 395.71 | TARGET | 22.11 |
| BUY | 2024-06-05 14:15:00 | 540.00 | 2024-08-01 10:15:00 | 604.90 | EXIT_EMA400 | 64.90 |
| BUY | 2024-07-22 09:15:00 | 639.00 | 2024-08-01 10:15:00 | 604.90 | EXIT_EMA400 | -34.10 |
| BUY | 2024-07-24 09:15:00 | 642.70 | 2024-08-01 10:15:00 | 604.90 | EXIT_EMA400 | -37.80 |
| SELL | 2024-09-16 09:15:00 | 546.80 | 2024-10-07 09:15:00 | 478.46 | TARGET | 68.34 |
| SELL | 2024-11-07 14:15:00 | 510.80 | 2024-11-11 09:15:00 | 479.97 | TARGET | 30.83 |
| SELL | 2024-11-04 14:15:00 | 500.35 | 2024-11-18 09:15:00 | 433.65 | TARGET | 66.70 |
| BUY | 2025-01-03 09:15:00 | 520.80 | 2025-01-06 09:15:00 | 499.00 | EXIT_EMA400 | -21.80 |
| SELL | 2025-01-22 10:15:00 | 466.30 | 2025-01-28 09:15:00 | 379.49 | TARGET | 86.81 |
| SELL | 2025-08-21 14:15:00 | 344.70 | 2025-09-15 09:15:00 | 346.00 | EXIT_EMA400 | -1.30 |
| SELL | 2025-08-21 15:15:00 | 341.15 | 2025-09-15 09:15:00 | 346.00 | EXIT_EMA400 | -4.85 |
| SELL | 2026-03-10 09:15:00 | 280.55 | 2026-04-16 10:15:00 | 282.59 | EXIT_EMA400 | -2.04 |
| SELL | 2026-04-07 13:15:00 | 271.68 | 2026-04-16 10:15:00 | 282.59 | EXIT_EMA400 | -10.91 |
