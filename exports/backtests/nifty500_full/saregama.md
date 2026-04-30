# Saregama India Ltd (SAREGAMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-05-31 09:15:00 → 2026-04-30 15:15:00 (5003 bars)
- **Last close:** 343.85
- **Strategy:** EMA 200/400 1H crossover
- **Target rule:** 1:3 RR (equity)
- **Stop-loss rule:** 1H close on wrong side of EMA400

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT3 | 7 |
| ENTRY1 | 7 |
| ENTRY2 | 5 |
| EXIT | 7 |

## P&L

- **Trades closed:** 12
- **Trades open at end:** 0
- **Winners / losers:** 2 / 10
- **Target hits / EMA400 exits:** 2 / 10
- **Total realized P&L (per unit):** -16.88
- **Avg P&L per closed trade:** -1.41

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 15:15:00 | 366.80 | 386.52 | 386.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 13:15:00 | 363.75 | 385.54 | 386.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-17 09:15:00 | 360.95 | 346.88 | 358.06 | EMA200 retest candle locked |

### Cycle 2 — BUY (started 2023-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 15:15:00 | 383.50 | 364.24 | 364.17 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 11:15:00 | 352.25 | 365.53 | 365.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 347.40 | 364.83 | 365.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-01 10:15:00 | 360.00 | 355.43 | 359.71 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-02-02 14:15:00 | 351.95 | 355.45 | 359.49 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 11:15:00 | 358.80 | 355.38 | 359.38 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2024-02-05 13:15:00 | 355.00 | 355.39 | 359.34 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2024-02-06 10:15:00 | 359.40 | 355.32 | 359.23 | Close above EMA400 |

### Cycle 4 — BUY (started 2024-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 10:15:00 | 415.20 | 361.58 | 361.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 09:15:00 | 425.70 | 377.79 | 370.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 13:15:00 | 386.40 | 387.10 | 376.93 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-03-05 14:15:00 | 392.80 | 387.16 | 377.01 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-03-11 11:15:00 | 376.70 | 386.79 | 377.69 | Close below EMA400 |

### Cycle 5 — SELL (started 2024-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 11:15:00 | 346.40 | 371.41 | 371.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-28 13:15:00 | 344.80 | 370.91 | 371.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 11:15:00 | 370.65 | 369.41 | 370.46 | EMA200 retest candle locked |

### Cycle 6 — BUY (started 2024-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 12:15:00 | 386.15 | 371.55 | 371.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 14:15:00 | 387.30 | 371.84 | 371.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-22 13:15:00 | 413.30 | 419.18 | 405.46 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2024-05-23 11:15:00 | 431.15 | 418.89 | 405.65 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2024-08-05 09:15:00 | 503.15 | 537.39 | 516.03 | Close below EMA400 |

### Cycle 7 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 510.65 | 534.87 | 534.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-06 09:15:00 | 505.25 | 532.78 | 533.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 504.65 | 499.08 | 512.73 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2024-11-29 09:15:00 | 488.20 | 498.98 | 512.22 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 508.00 | 498.92 | 511.93 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2024-12-02 09:15:00 | 516.45 | 499.27 | 511.91 | Close above EMA400 |

### Cycle 8 — BUY (started 2025-01-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-10 09:15:00 | 531.70 | 509.48 | 509.39 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 15:15:00 | 488.75 | 511.19 | 511.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-30 09:15:00 | 484.45 | 510.93 | 511.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 10:15:00 | 522.60 | 506.74 | 508.96 | EMA200 retest candle locked |

### Cycle 10 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 535.40 | 511.15 | 511.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-10 14:15:00 | 540.45 | 513.39 | 512.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-11 09:15:00 | 509.50 | 513.46 | 512.28 | EMA200 retest candle locked |

### Cycle 11 — SELL (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-17 14:15:00 | 477.50 | 511.44 | 511.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 09:15:00 | 470.20 | 510.68 | 511.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-07 12:15:00 | 500.35 | 491.32 | 499.55 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-03-10 09:15:00 | 486.20 | 491.34 | 499.39 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 486.20 | 491.34 | 499.39 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-12 09:15:00 | 479.00 | 490.75 | 498.54 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 495.00 | 490.15 | 497.74 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-03-17 09:15:00 | 491.80 | 490.16 | 497.71 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 495.30 | 490.26 | 497.68 | EMA400 retest candle locked |
| Exit — 1H close above EMA400 | 2025-03-17 13:15:00 | 498.15 | 490.41 | 497.68 | Close above EMA400 |

### Cycle 12 — BUY (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 14:15:00 | 514.65 | 502.19 | 502.19 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 466.60 | 502.00 | 502.11 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 531.00 | 502.09 | 502.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 533.95 | 504.02 | 503.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 524.15 | 528.37 | 517.71 | EMA200 retest candle locked |
| First Entry (BUY) — break of retest1 high, sustain | 2025-05-05 13:15:00 | 537.45 | 528.50 | 518.44 | Buy entry 1 (retest1 break) |
| Exit — 1H close below EMA400 | 2025-06-13 09:15:00 | 537.65 | 549.24 | 538.49 | Close below EMA400 |

### Cycle 15 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 503.50 | 532.08 | 532.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 14:15:00 | 502.20 | 531.23 | 531.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 518.20 | 506.28 | 516.20 | EMA200 retest candle locked |
| First Entry (SELL) — break of retest1 low, sustain | 2025-07-25 10:15:00 | 486.00 | 504.59 | 513.71 | Sell entry 1 (retest1 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 492.95 | 494.04 | 505.54 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-07 09:15:00 | 480.55 | 493.82 | 505.03 | Sell entry 2 (retest2 break) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 497.30 | 489.23 | 499.04 | EMA400 retest candle locked |
| Second Entry (SELL) — break of retest2 low, sustain | 2025-08-26 14:15:00 | 475.95 | 488.99 | 498.04 | Sell entry 2 (retest2 break) |
| Exit — 1H close above EMA400 | 2025-09-02 09:15:00 | 499.75 | 487.63 | 496.32 | Close above EMA400 |


## Closed Trades

| Trend | Entry Time | Entry | Exit Time | Exit | Reason | P&L |
|-------|-----------|-------|-----------|------|--------|-----|
| SELL | 2024-02-02 14:15:00 | 351.95 | 2024-02-06 10:15:00 | 359.40 | EXIT_EMA400 | -7.45 |
| SELL | 2024-02-05 13:15:00 | 355.00 | 2024-02-06 10:15:00 | 359.40 | EXIT_EMA400 | -4.40 |
| BUY | 2024-03-05 14:15:00 | 392.80 | 2024-03-11 11:15:00 | 376.70 | EXIT_EMA400 | -16.10 |
| BUY | 2024-05-23 11:15:00 | 431.15 | 2024-05-30 14:15:00 | 507.64 | TARGET | 76.49 |
| SELL | 2024-11-29 09:15:00 | 488.20 | 2024-12-02 09:15:00 | 516.45 | EXIT_EMA400 | -28.25 |
| SELL | 2025-03-10 09:15:00 | 486.20 | 2025-03-17 13:15:00 | 498.15 | EXIT_EMA400 | -11.95 |
| SELL | 2025-03-12 09:15:00 | 479.00 | 2025-03-17 13:15:00 | 498.15 | EXIT_EMA400 | -19.15 |
| SELL | 2025-03-17 09:15:00 | 491.80 | 2025-03-17 13:15:00 | 498.15 | EXIT_EMA400 | -6.35 |
| BUY | 2025-05-05 13:15:00 | 537.45 | 2025-06-04 09:15:00 | 594.47 | TARGET | 57.02 |
| SELL | 2025-07-25 10:15:00 | 486.00 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -13.75 |
| SELL | 2025-08-07 09:15:00 | 480.55 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -19.20 |
| SELL | 2025-08-26 14:15:00 | 475.95 | 2025-09-02 09:15:00 | 499.75 | EXIT_EMA400 | -23.80 |
